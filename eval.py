import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    cost_log = []  # one (goal_keys, costs) entry per CEM replan call
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)

        # -- log CEM costs per replan (wrapper only, package untouched) --
        _orig_solve = solver.solve

        def _logged_solve(info_dict, init_action=None):
            out = _orig_solve(info_dict, init_action=init_action)
            key = info_dict.get("goal_proprio", info_dict.get("goal_state"))
            if torch.is_tensor(key):
                key = key.detach().cpu().numpy()
            key = np.asarray(key, dtype=np.float64).reshape(len(key), -1)
            cost_log.append((key, np.asarray(out["costs"], dtype=np.float64)))
            return out

        solver.solve = _logged_solve

        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()

    print(metrics)

    # -- per-episode report --
    successes = np.asarray(metrics["episode_successes"], dtype=bool)
    n = len(successes)
    cost_p1 = np.full(n, np.nan)
    cost_p2 = np.full(n, np.nan)

    if cost_log:
        keys0, cost_p1[:] = cost_log[0]
        assert len(keys0) == n, "first replan should cover all envs"
        for keys_k, costs_k in cost_log[1:]:
            d = ((keys_k[:, None, :] - keys0[None, :, :]) ** 2).sum(-1)
            match = d.argmin(1)
            if d.min(1).max() > 1e-8:
                print("WARNING: inexact goal-key match in cost log")
            cost_p2[match] = costs_k

    report_lines = []
    for i in range(n):
        s = "SUCCESS" if successes[i] else "FAIL   "
        c1 = f"{cost_p1[i]:.3f}" if not np.isnan(cost_p1[i]) else " — "
        c2 = f"{cost_p2[i]:.3f}" if not np.isnan(cost_p2[i]) else " — "
        report_lines.append(
            f"ep {i:03d} | {s} | cost_plan1={c1} | cost_plan2={c2} "
            f"| dataset_ep={eval_episodes[i]} | video=env_{i}.mp4"
        )
    report = "\n".join(report_lines)
    print(report)

    early = successes & np.isnan(cost_p2)   # done before t=25
    late = successes & ~np.isnan(cost_p2)
    fail = ~successes

    def _stat(a):
        a = a[~np.isnan(a)]
        return f"mean={a.mean():.3f} median={np.median(a):.3f} (n={len(a)})" if len(a) else "n/a"

    summary = "\n".join([
        f"success before 2nd replan: {early.sum()}",
        f"success after 2nd replan:  {late.sum()}",
        f"failures:                  {fail.sum()}",
        f"cost_plan2 | late success: {_stat(cost_p2[late])}",
        f"cost_plan2 | failures:     {_stat(cost_p2[fail])}",
    ])
    print(summary)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"per-episode:\n{report}\n")
        f.write(f"summary:\n{summary}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()