import numpy as np
import torch
from tqdm import tqdm


class SampleGenerator:
    def __init__(self, cfg):
        self.max_interval_sum = cfg.max_interval_sum
        self.batch_size = cfg.batch_size
        self.pool_ceiling = cfg.batch_size * cfg.pool_ceiling_multiplier
        self.p = cfg.interval_dist_p
        self.rng = np.random.default_rng(cfg.seed)
        self.active_pools = [{"intervals": (1, 1, 1), "samples": []}]
        self.stride = getattr(cfg, "stride", 1)

    def _sample_intervals(self):
        return tuple(max(1, int(self.rng.geometric(self.p))) for _ in range(3))

    def create_pool(self):
        intervals = self._sample_intervals()
        if sum(intervals) > self.max_interval_sum:
            return self.create_pool()
        return {"intervals": intervals, "samples": []}

    def get_ready_batches(self):
        drained = []
        for pool_id in range(len(self.active_pools) - 1, -1, -1):
            pool = self.active_pools[pool_id]
            if len(pool["samples"]) >= self.pool_ceiling:
                self.active_pools.pop(pool_id)
                self.rng.shuffle(pool["samples"])
                num_batches = len(pool["samples"]) // self.batch_size
                batches = [
                    pool["samples"][i * self.batch_size : (i + 1) * self.batch_size]
                    for i in range(num_batches)
                ]
                config = pool["intervals"]
                drained.append((config, batches))
                if pool_id == 0:
                    self.active_pools.insert(0, {"intervals": (1, 1, 1), "samples": []})
                else:
                    self.active_pools.append(self.create_pool())
        return drained

    def extract_episode(self, episode_id, episode_length):
        for pool in self.active_pools:
            intervals = pool["intervals"]
            span = sum(intervals)
            start_idx = 0
            while start_idx + span < episode_length:
                pool["samples"].append((episode_id, start_idx, intervals))
                start_idx += self.stride


def precompute_batches(lance_dataset, cfg_data_sampling, ep_ids, seed, desc="Precomputing"):
    """Single-threaded precomputation. Returns np.ndarray of shape (num_batches, batch_size, 5)."""
    gen = SampleGenerator(cfg_data_sampling)
    for _ in range(cfg_data_sampling.num_pools - 1):
        gen.active_pools.append(gen.create_pool())

    episode_lengths = {i: int(l) for i, l in enumerate(lance_dataset.lengths)}
    rng = np.random.default_rng(seed)
    shuffled_ep_ids = list(ep_ids)
    rng.shuffle(shuffled_ep_ids)

    all_batches = []

    for ep_id in tqdm(shuffled_ep_ids, desc=desc, total=len(shuffled_ep_ids)):
        gen.extract_episode(ep_id, episode_lengths[ep_id])
        ready = gen.get_ready_batches()
        for config, batches in ready:
            all_batches.extend(batches)

    # Drain remaining pools
    for pool in gen.active_pools:
        if len(pool["samples"]) >= cfg_data_sampling.batch_size:
            rng.shuffle(pool["samples"])
            num_batches = len(pool["samples"]) // cfg_data_sampling.batch_size
            for i in range(num_batches):
                all_batches.append(
                    pool["samples"][i * cfg_data_sampling.batch_size : (i + 1) * cfg_data_sampling.batch_size]
                )

    if not all_batches:
        return np.zeros((0, cfg_data_sampling.batch_size, 5), dtype=np.int32)

    # Convert to compact numpy array: (num_batches, batch_size, 5)
    # 5 = [ep_id, start_idx, interval_0, interval_1, interval_2]
    num_batches = len(all_batches)
    batch_size = cfg_data_sampling.batch_size
    arr = np.zeros((num_batches, batch_size, 5), dtype=np.int32)

    for b_idx, batch in enumerate(all_batches):
        for s_idx, (ep_id, start_idx, intervals) in enumerate(batch):
            arr[b_idx, s_idx, 0] = ep_id
            arr[b_idx, s_idx, 1] = start_idx
            arr[b_idx, s_idx, 2] = intervals[0]
            arr[b_idx, s_idx, 3] = intervals[1]
            arr[b_idx, s_idx, 4] = intervals[2]

    return arr


class SampleLoader:
    def __init__(self, lance_dataset, transform=None, action_mean=None, action_std=None,
                 episode_lengths=None):
        self.lance = lance_dataset
        self.fs = lance_dataset.frameskip
        self.transform = transform
        self.action_mean = action_mean
        self.action_std = action_std
        self.episode_lengths = episode_lengths or {}

    def load_batch(self, batch_arr):
        """batch_arr: np.ndarray of shape (batch_size, 5)"""
        all_pixels = []
        all_action_lists = [[] for _ in range(3)]
        all_img_times = []
        skipped = 0

        for s_idx in range(batch_arr.shape[0]):
            ep_id = int(batch_arr[s_idx, 0])
            start_idx = int(batch_arr[s_idx, 1])
            intervals = (
                int(batch_arr[s_idx, 2]),
                int(batch_arr[s_idx, 3]),
                int(batch_arr[s_idx, 4]),
            )

            img_ids = [start_idx + sum(intervals[:i]) for i in range(4)]
            act_ids = [list(range(img_ids[i], img_ids[i + 1])) for i in range(3)]

            start_obs = img_ids[0]
            end_obs = img_ids[-1] + 1
            start_raw = start_obs * self.fs

            ep_obs_len = self.episode_lengths.get(ep_id, end_obs)
            end_raw = min(end_obs * self.fs, ep_obs_len * self.fs)

            if start_raw >= end_raw:
                skipped += 1
                continue

            # === ROBUST LOAD: try full range, then single obs, then skip ===
            raw = None
            try:
                raw = self.lance._load_slice(ep_id, start_raw, end_raw)
            except RuntimeError:
                # Try loading just the first observation as fallback
                try:
                    raw = self.lance._load_slice(ep_id, start_raw, start_raw + self.fs)
                except RuntimeError:
                    skipped += 1
                    continue

            all_imgs = raw["pixels"]
            rel_img_ids = [i - start_obs for i in img_ids]

            # Defensive: if loaded data is shorter than expected, skip
            if not all_imgs.shape[0] or max(rel_img_ids) >= all_imgs.shape[0]:
                skipped += 1
                continue

            pixels = all_imgs[rel_img_ids]
            all_pixels.append(pixels)

            all_actions = raw["action"]
            for i, chunk in enumerate(act_ids):
                raw_start = (chunk[0] - start_obs) * self.fs
                raw_end = (chunk[-1] - start_obs + 1) * self.fs
                raw_end = min(raw_end, all_actions.shape[0])
                if raw_start >= raw_end:
                    continue
                raw_actions = all_actions[raw_start:raw_end]

                if self.action_mean is not None and self.action_std is not None:
                    mean = self.action_mean.to(raw_actions.device)
                    std = self.action_std.to(raw_actions.device)
                    raw_actions = (raw_actions - mean) / std

                L = len(chunk)
                if raw_actions.shape[0] < L * self.fs:
                    continue
                obs_actions = raw_actions.reshape(L, self.fs * raw_actions.shape[-1])
                all_action_lists[i].append(obs_actions)

            img_times = [0]
            for chunk in act_ids[:-1]:
                img_times.append(img_times[-1] + len(chunk))

            all_img_times.append(img_times)

        if not all_pixels:
            raise RuntimeError("load_batch: all samples were skipped")

        if skipped > 0:
            print(f"[WARN] load_batch skipped {skipped}/{batch_arr.shape[0]} bad samples")

        batch = {
            "pixels": torch.stack(all_pixels),
            "action": [torch.stack(chunks) for chunks in all_action_lists],
            "img_times": torch.tensor(all_img_times),
        }

        if self.transform:
            batch = self.transform(batch)

        return batch


class VariableHorizonDataset(torch.utils.data.IterableDataset):
    def __init__(self, lance_dataset, cfg_data_sampling, transform=None, seed=0, ep_ids=None,
                 action_mean=None, action_std=None, precomputed_batches=None):
        self.lance = lance_dataset
        self.cfg = cfg_data_sampling
        self.transform = transform
        self.seed = seed
        self.episode_lengths = {i: int(l) for i, l in enumerate(lance_dataset.lengths)}
        self.all_ep_ids = ep_ids if ep_ids is not None else list(self.episode_lengths.keys())
        self.action_mean = action_mean
        self.action_std = action_std
        self.precomputed_batches = precomputed_batches

    def __len__(self):
        if self.precomputed_batches is not None:
            return self.precomputed_batches.shape[0]
        avg_interval = max(1, int(1.0 / self.cfg.interval_dist_p))
        typical_span = avg_interval * 3
        total_samples = sum(
            max(1, self.episode_lengths[i] - typical_span)
            for i in self.all_ep_ids
        )
        return max(1, total_samples // self.cfg.batch_size)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if self.precomputed_batches is not None:
            batches = self.precomputed_batches
            if worker_info is not None:
                per_worker = len(batches) // worker_info.num_workers
                start = worker_info.id * per_worker
                end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(batches)
                batches = batches[start:end]

            loader = SampleLoader(
                self.lance, self.transform, self.action_mean, self.action_std,
                episode_lengths=self.episode_lengths
            )
            for b_idx in range(batches.shape[0]):
                yield loader.load_batch(batches[b_idx])
            return

        # On-the-fly fallback
        if worker_info is None:
            gen = SampleGenerator(self.cfg)
            for _ in range(self.cfg.num_pools - 1):
                gen.active_pools.append(gen.create_pool())
            loader = SampleLoader(
                self.lance, self.transform, self.action_mean, self.action_std,
                episode_lengths=self.episode_lengths
            )
            yield from self._iterate(gen, loader, self.all_ep_ids)
        else:
            per_worker = len(self.all_ep_ids) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = (
                start + per_worker
                if worker_info.id < worker_info.num_workers - 1
                else len(self.all_ep_ids)
            )
            worker_ep_ids = self.all_ep_ids[start:end]

            gen = SampleGenerator(self.cfg)
            for _ in range(self.cfg.num_pools - 1):
                gen.active_pools.append(gen.create_pool())
            loader = SampleLoader(
                self.lance, self.transform, self.action_mean, self.action_std,
                episode_lengths=self.episode_lengths
            )
            yield from self._iterate(gen, loader, worker_ep_ids)

    def _iterate(self, gen, loader, ep_ids):
        rng = np.random.default_rng(self.seed)
        rng.shuffle(ep_ids)

        for ep_id in ep_ids:
            gen.extract_episode(ep_id, self.episode_lengths[ep_id])
            ready = gen.get_ready_batches()
            for config, batches in ready:
                for batch_samples in batches:
                    yield loader.load_batch(batch_samples)


def print_batch_distribution(batch_arr, desc="Train"):
    """Print horizon distribution of precomputed batches."""
    if batch_arr.shape[0] == 0:
        print(f"[{desc}] No batches to analyze.")
        return

    # batch_arr: (num_batches, batch_size, 5)
    # intervals at indices 2,3,4
    horizons = batch_arr[:, :, 2:5].sum(axis=2)  # (num_batches, batch_size)
    batch_horizons = horizons[:, 0]  # all samples in batch same pool
    total_batches = len(batch_horizons)

    print(f"\n{'='*60}")
    print(f"[{desc}] Batch horizon distribution (total: {total_batches})")
    print(f"{'='*60}")

    buckets = [
        (3, 3, "exactly 3  (1,1,1)"),
        (4, 6, "4-6"),
        (7, 10, "7-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 100, "51-100"),
        (101, 150, "101-150"),
        (151, 200, "151-200"),
    ]

    for low, high, label in buckets:
        count = ((batch_horizons >= low) & (batch_horizons <= high)).sum()
        pct = 100 * count / total_batches
        bar = "█" * int(pct / 2)
        print(f"  horizon {label:15s}: {count:6d} ({pct:5.1f}%) {bar}")

    print(f"\n  Exact small horizon breakdown:")
    for h in range(3, 11):
        count = (batch_horizons == h).sum()
        if count > 0:
            pct = 100 * count / total_batches
            print(f"    horizon={h}: {count:6d} ({pct:5.1f}%)")

    # Most common interval combinations
    print(f"\n  Most common interval configs (top 10):")
    configs = {}
    for b in range(batch_arr.shape[0]):
        iv = tuple(batch_arr[b, 0, 2:5].tolist())
        configs[iv] = configs.get(iv, 0) + 1

    top_configs = sorted(configs.items(), key=lambda x: x[1], reverse=True)[:10]
    for iv, count in top_configs:
        pct = 100 * count / total_batches
        print(f"    {iv}: {count:6d} ({pct:5.1f}%)")

    print(f"{'='*60}\n")