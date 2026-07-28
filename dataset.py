import io
import random
from typing import Optional

import lance
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import IterableDataset


def decode_jpeg_bytes(jpeg_bytes):
    img = Image.open(io.BytesIO(jpeg_bytes))
    img = img.convert("RGB")
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return img


class VariableHorizonDataset(IterableDataset):
    """Variable-horizon window dataset over a Lance file.

    Key properties:
    - Lance is only ever read SEQUENTIALLY, one full episode at a time.
      All windows of that episode are decoded immediately (while the data
      is hot) and the shuffle buffer stores actual samples, not metadata.
    - Phase-offset sampling: each window draws a phase p in [0, frameskip-1]
      and starts at raw frame start*frameskip + p. Action bundles are
      relative to the phase. This matches the original le-wm sampling,
      which allowed window starts at ANY raw frame (~5x more distinct
      windows per episode than bundle-aligned starts).
    - time_ids stay in BUNDLED units; train.py makes them window-relative.
    """

    def __init__(
        self,
        lance_path: str,
        num_steps: int = 4,
        windows_per_episode_factor: float = 1.0,
        max_gap: int = 50,
        geometric_p: float = 0.5,
        gap_sampling: str = "geometric",   # geometric | fixed | uniform
        fixed_gap: int = 1,

        buffer_size: int = 1000,
        frameskip: int = 5,
        transform: Optional[callable] = None,
    ):
        super().__init__()
        self.lance_path = lance_path
        self.num_steps = num_steps
        self.num_gaps = num_steps - 1
        self.windows_per_episode_factor = windows_per_episode_factor
        self.max_gap = max_gap
        self.geometric_p = geometric_p
        self.gap_sampling = gap_sampling
        self.fixed_gap = fixed_gap

        self.buffer_size = buffer_size
        self.frameskip = frameskip
        self.transform = transform

        self._scan_episodes()
        self.episode_order = list(range(self.num_episodes))
        self._col_cache = {}
        self._ds = None  # Lance handle, opened lazily once per worker

    # ---------------- Lance access ----------------

    def _get_ds(self):
        """One Lance handle per process/worker (not picklable, opened lazily)."""
        if self._ds is None:
            self._ds = lance.dataset(self.lance_path)
        return self._ds

    def _scan_episodes(self):
        ds = lance.dataset(self.lance_path)
        table = ds.to_table(columns=["episode_idx"])
        episode_indices = table["episode_idx"].to_numpy()

        boundaries = np.where(np.diff(episode_indices) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(episode_indices)]])

        self.episode_starts = starts
        self.episode_ends = ends
        self.episode_lengths = ends - starts
        self.num_episodes = len(starts)

    def _load_episode(self, episode_idx):
        """Single sequential read of one full episode. Returns RAW actions."""
        ds = self._get_ds()
        start = self.episode_starts[episode_idx]
        length = self.episode_lengths[episode_idx]

        scanner = ds.scanner(offset=start, limit=length)
        table = scanner.to_table()

        raw_actions = np.stack(table["action"].to_numpy())  # (T_raw, 2)
        pixels = table["pixels"].to_numpy()
        proprio = np.stack(table["proprio"].to_numpy()) if "proprio" in table.column_names else None
        state = np.stack(table["state"].to_numpy()) if "state" in table.column_names else None

        return raw_actions, pixels, proprio, state

    # ---------------- normalizer interface ----------------

    def get_col_data(self, col):
        if col not in self._col_cache:
            ds = lance.dataset(self.lance_path)
            table = ds.to_table(columns=[col])
            data = np.stack(table[col].to_numpy())
            if col == "action":
                n = len(data) // self.frameskip
                data = data[:n * self.frameskip].reshape(n, self.frameskip * 2)
            self._col_cache[col] = data
        return self._col_cache[col]

    def get_dim(self, col):
        if col == "action":
            return self.frameskip * 2
        elif col == "proprio":
            return 4
        elif col == "state":
            return 7
        return 0

    # ---------------- window sampling ----------------

    def _sample_gaps(self):
        if self.gap_sampling == "fixed":
            return np.full(self.num_gaps, self.fixed_gap)
        if self.gap_sampling == "uniform":
            return np.random.randint(1, self.max_gap + 1, size=self.num_gaps)
        # default: geometric (unchanged legacy behavior)
        gaps = np.random.geometric(self.geometric_p, size=self.num_gaps)
        gaps = np.clip(gaps, 1, self.max_gap)
        return gaps

    def _sample_window_params(self, episode_idx):
        """Draw (start_offset, gaps, phase) tuples for one episode."""
        bundled_len = self.episode_lengths[episode_idx] // self.frameskip
        n_windows = max(1, int(bundled_len * self.windows_per_episode_factor))
        params = []
        for _ in range(n_windows):
            gaps = self._sample_gaps()
            total_span = gaps.sum()
            if total_span >= bundled_len:
                continue
            start_offset = random.randint(0, bundled_len - total_span - 1)
            phase = random.randint(0, self.frameskip - 1)
            params.append((start_offset, gaps, phase))
        return params

    def _materialize_window(self, ep_data, start_offset, gaps, phase):
        """Decode + transform one window from already-loaded episode data."""
        raw_actions, pixels, proprio, state = ep_data

        # bundled-unit time ids (train.py makes them window-relative)
        time_ids = [start_offset]
        for g in gaps:
            time_ids.append(time_ids[-1] + g)
        time_ids = np.array(time_ids, dtype=np.int64)

        # raw frame indices of the observations, phase-shifted
        raw_obs_ids = time_ids * self.frameskip + phase

        obs_pixels = [decode_jpeg_bytes(pixels[t]) for t in raw_obs_ids]
        obs_pixels = torch.stack(obs_pixels)

        obs_proprio = proprio[raw_obs_ids] if proprio is not None else None
        obs_state = state[raw_obs_ids] if state is not None else None

        # raw actions spanning the whole window, bundled relative to phase:
        # raw_obs_ids[-1] - raw_obs_ids[0] = span * frameskip, divisible by frameskip
        flat = raw_actions[raw_obs_ids[0]:raw_obs_ids[-1]]  # (span*frameskip, 2)
        all_bundled = torch.from_numpy(
            flat.reshape(-1, self.frameskip * 2)
        ).float()  # (span, frameskip*2)

        chunk_boundaries = []
        for i in range(self.num_gaps):
            cs = time_ids[i] - time_ids[0]
            ce = time_ids[i + 1] - time_ids[0]
            chunk_boundaries.append((cs, ce))

        sample = {
            "pixels": obs_pixels,
            "action": all_bundled,
            "time_ids": time_ids,
        }
        if obs_proprio is not None:
            sample["proprio"] = torch.from_numpy(obs_proprio).float()
        if obs_state is not None:
            sample["state"] = torch.from_numpy(obs_state).float()

        if self.transform is not None:
            sample = self.transform(sample)

        normed = sample["action"]
        action_chunks = []
        action_lengths = []
        for cs, ce in chunk_boundaries:
            chunk = normed[cs:ce].numpy()
            action_chunks.append(chunk)
            action_lengths.append(len(chunk))

        sample["action_chunks"] = action_chunks
        sample["action_lengths"] = action_lengths

        return sample

    # ---------------- iteration ----------------

    def __len__(self):
        total = 0
        for length in self.episode_lengths:
            bundled_len = length // self.frameskip
            total += max(1, int(bundled_len * self.windows_per_episode_factor))
        return total

    def __iter__(self):
        # Shard episodes across workers so each window is yielded once per epoch
        worker_info = torch.utils.data.get_worker_info()
        episode_order = self.episode_order.copy()
        random.shuffle(episode_order)
        if worker_info is not None:
            episode_order = episode_order[worker_info.id :: worker_info.num_workers]

        buffer = []
        for episode_idx in episode_order:
            params = self._sample_window_params(episode_idx)
            if not params:
                continue

            # ONE sequential Lance read per episode
            ep_data = self._load_episode(episode_idx)

            random.shuffle(params)
            for start_offset, gaps, phase in params:
                # Decode while the episode is hot; buffer stores the SAMPLE
                buffer.append(self._materialize_window(ep_data, start_offset, gaps, phase))
                if len(buffer) >= self.buffer_size:
                    idx = random.randint(0, len(buffer) - 1)
                    yield buffer.pop(idx)

        random.shuffle(buffer)
        for sample in buffer:
            yield sample

    # ---------------- pickling for workers ----------------

    def __getstate__(self):
        """Clear heavy caches before pickling for worker processes."""
        state = self.__dict__.copy()
        state["_col_cache"] = {}
        state["_ds"] = None  # Lance handle is not picklable; reopened lazily
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._col_cache = {}
        self._ds = None


def _to_tensor(x):
    return x if torch.is_tensor(x) else torch.from_numpy(x)


def variable_horizon_collate(batch):
    B = len(batch)
    num_obs = len(batch[0]["time_ids"])
    num_chunks = num_obs - 1

    pixels = torch.stack([s["pixels"] for s in batch])

    max_len = max(
        max(len(chunk) for chunk in s["action_chunks"]) for s in batch
    )
    action_dim = batch[0]["action_chunks"][0].shape[-1]

    action_padded = torch.zeros(B, num_chunks, max_len, action_dim)
    action_lengths = torch.zeros(B, num_chunks, dtype=torch.long)

    for i, sample in enumerate(batch):
        for j, chunk in enumerate(sample["action_chunks"]):
            L = len(chunk)
            action_padded[i, j, :L] = _to_tensor(chunk)
            action_lengths[i, j] = L

    time_ids = torch.stack([_to_tensor(s["time_ids"]) for s in batch])

    out = {
        "pixels": pixels,
        "action": action_padded,
        "time_ids": time_ids,
        "action_lengths": action_lengths,
    }

    if "proprio" in batch[0]:
        out["proprio"] = torch.stack([_to_tensor(s["proprio"]) for s in batch])
    if "state" in batch[0]:
        out["state"] = torch.stack([_to_tensor(s["state"]) for s in batch])

    return out
