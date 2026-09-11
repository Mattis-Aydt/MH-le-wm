import io
import random
from typing import Optional, Sequence, Union

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
    """Fixed-gap enumeration dataset over a Lance file.

    Semantics (exact original le-wm equivalence):
    - For every episode, for every gap g in fixed_gaps, EVERY valid window is
      enumerated: all bundle starts where the window fits, times all phases
      in [0, frameskip-1]. With fixed_gaps=[1] this yields exactly the window
      set of the original le-wm dataset (a window starting at every raw
      frame). fixed_gaps=[1, 2, 3] is the union of those homogeneous sets.
    - Lance is only ever read SEQUENTIALLY, one full episode at a time.
      All windows of that episode are decoded immediately (while the data
      is hot) and the shuffle buffer stores actual samples, not metadata.
    - time_ids stay in BUNDLED units; train.py makes them window-relative.
    """

    def __init__(
        self,
        lance_path: str,
        num_steps: int = 4,
        fixed_gaps: Union[int, Sequence[int]] = (1,),
        buffer_size: int = 1000,
        frameskip: int = 5,
        transform: Optional[callable] = None,
    ):
        super().__init__()
        self.lance_path = lance_path
        self.num_steps = num_steps
        self.num_gaps = num_steps - 1
        if isinstance(fixed_gaps, int):
            fixed_gaps = [fixed_gaps]
        self.fixed_gaps = [int(g) for g in fixed_gaps]
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

    # ---------------- window enumeration ----------------

    def _windows_per_episode(self, bundled_len):
        """Exact window count of one episode over all gap classes."""
        return sum(
            max(0, bundled_len - g * self.num_gaps) * self.frameskip
            for g in self.fixed_gaps
        )

    def _episode_windows(self, episode_idx):
        """Enumerate ALL (start, gap, phase) windows of one episode.

        A window starting at bundle `start` with phase `phase` observes raw
        frames start*frameskip+phase, (start+g)*frameskip+phase, ... — so
        gap=1 with all phases covers exactly the raw-frame starts of the
        original le-wm sampling.
        """
        bundled_len = self.episode_lengths[episode_idx] // self.frameskip
        params = []
        for g in self.fixed_gaps:
            span = g * self.num_gaps
            for start in range(bundled_len - span):
                for phase in range(self.frameskip):
                    params.append((start, g, phase))
        return params

    def _materialize_window(self, ep_data, start, gap, phase):
        """Decode + transform one window from already-loaded episode data."""
        raw_actions, pixels, proprio, state = ep_data

        # bundled-unit time ids (train.py makes them window-relative)
        time_ids = start + gap * np.arange(self.num_steps, dtype=np.int64)

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
        # homogeneous gaps: chunk j = bundles [j*gap, (j+1)*gap)
        action_chunks = [
            normed[j * gap:(j + 1) * gap].numpy() for j in range(self.num_gaps)
        ]
        action_lengths = [gap] * self.num_gaps

        sample["action_chunks"] = action_chunks
        sample["action_lengths"] = action_lengths

        return sample

    # ---------------- iteration ----------------

    def __len__(self):
        """Exact window count over the ACTIVE episode set (train/val split aware)."""
        return sum(self._windows_per_episode(self.episode_lengths[ep] // self.frameskip)
                   for ep in self.episode_order)

    def __iter__(self):
        # Shard episodes across workers so each window is yielded once per epoch
        worker_info = torch.utils.data.get_worker_info()
        episode_order = self.episode_order.copy()
        random.shuffle(episode_order)
        if worker_info is not None:
            episode_order = episode_order[worker_info.id :: worker_info.num_workers]

        buffer = []
        for episode_idx in episode_order:
            params = self._episode_windows(episode_idx)
            if not params:
                continue

            # ONE sequential Lance read per episode
            ep_data = self._load_episode(episode_idx)

            random.shuffle(params)
            for start, gap, phase in params:
                # Decode while the episode is hot; buffer stores the SAMPLE
                buffer.append(self._materialize_window(ep_data, start, gap, phase))
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