import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from data_sampling import VariableHorizonDataset, precompute_batches, print_batch_distribution
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback
import torch.nn.functional as F


class RecomputeBatchesCallback(pl.Callback):
    """Recompute dataset batches at the start of each epoch with a new seed."""

    def __init__(self, train_dataset, val_dataset, cfg_data_sampling, train_ep_ids, val_ep_ids):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg_data_sampling = cfg_data_sampling
        self.train_ep_ids = train_ep_ids
        self.val_ep_ids = val_ep_ids
        self.epoch = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch += 1
        seed = self.train_dataset.seed + self.epoch
        print(f"\n[Recompute] Epoch {self.epoch}: recomputing train batches with seed {seed}...")
        self.train_dataset.precomputed_batches = precompute_batches(
            self.train_dataset.lance,
            self.cfg_data_sampling,
            self.train_ep_ids,
            seed,
            desc=f"Epoch {self.epoch} train"
        )
        print(f"[Recompute] Train batches: {len(self.train_dataset.precomputed_batches)}")

        val_seed = self.val_dataset.seed + self.epoch
        print(f"[Recompute] Recomputing val batches with seed {val_seed}...")
        self.val_dataset.precomputed_batches = precompute_batches(
            self.val_dataset.lance,
            self.cfg_data_sampling,
            self.val_ep_ids,
            val_seed,
            desc=f"Epoch {self.epoch} val"
        )
        print(f"[Recompute] Val batches: {len(self.val_dataset.precomputed_batches)}\n")

def lejepa_forward(self, batch, stage, cfg):
    ctx_len = cfg.history_size
    lambd = cfg.loss.sigreg.weight

    for i in range(len(batch["action"])):
        batch["action"][i] = torch.nan_to_num(batch["action"][i], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    # === Collapse detection: pairwise cosine similarity of chunk latents ===
    # Healthy: ~0.0-0.3 (diverse). Collapse: → 1.0 (all identical).
    if act_emb is not None and act_emb.numel() > 0:
        B, T, D = act_emb.shape
        if B * T > 1:
            flat = F.normalize(act_emb.reshape(-1, D), dim=-1)
            sim = flat @ flat.t()
            mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
            if mask.any():
                self.log(f"{stage}/act_emb_cos_sim", sim[mask].mean().detach(), on_step=True, sync_dist=True)

    ctx_emb = emb[:, :-1]
    tgt_emb = emb[:, 1:]

    pred_emb = self.model.predict(ctx_emb, act_emb, batch["img_times"])

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb[:, :-1].transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    # Compute action normalization stats from raw dataset
    action_col = torch.from_numpy(np.array(dataset.get_col_data("action")))
    action_col = action_col[~torch.isnan(action_col).any(dim=1)]
    action_mean = action_col.mean(0, keepdim=True)  # (1, 2)
    action_std = action_col.std(0, keepdim=True)    # (1, 2)

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col == "action":
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)

    # Split episodes
    all_ep_ids = list(range(len(dataset.lengths)))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_ep_ids)
    n_train = int(len(all_ep_ids) * cfg.train_split)
    train_ep_ids = all_ep_ids[:n_train]
    val_ep_ids = all_ep_ids[n_train:]

    #########################
    ##  precompute batches ##
    #########################

    print("Precomputing training batches...")
    train_batches = precompute_batches(
        dataset, cfg.data.data_sampling, train_ep_ids, cfg.seed, desc="Precomputing train"
    )
    print(f"Train batches: {len(train_batches)}")
    print_batch_distribution(train_batches, desc="Train")

    print("Precomputing validation batches...")
    val_batches = precompute_batches(
        dataset, cfg.data.data_sampling, val_ep_ids, cfg.seed + 1, desc="Precomputing val"
    )
    print(f"Val batches: {len(val_batches)}")
    print_batch_distribution(val_batches, desc="Val")

    # Exact steps for scheduler
    batches_per_epoch = len(train_batches)
    total_steps = batches_per_epoch * cfg.trainer.max_epochs
    warmup_steps = max(1, int(0.01 * total_steps))
    print(f"Exact steps per epoch: {batches_per_epoch}, total: {total_steps}, warmup: {warmup_steps}")

    # Create datasets
    train_dataset = VariableHorizonDataset(
        dataset, cfg.data.data_sampling, transform=transform, seed=cfg.seed,
        ep_ids=train_ep_ids, action_mean=action_mean, action_std=action_std,
        precomputed_batches=train_batches,
    )
    val_dataset = VariableHorizonDataset(
        dataset, cfg.data.data_sampling, transform=transform, seed=cfg.seed + 1,
        ep_ids=val_ep_ids, action_mean=action_mean, action_std=action_std,
        precomputed_batches=val_batches,
    )

    # DataLoaders
    train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.loader.get("prefetch_factor", 2),
        pin_memory=cfg.loader.get("pin_memory", True),
    )
    val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.loader.get("prefetch_factor", 2),
        pin_memory=cfg.loader.get("pin_memory", True),
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": total_steps,
            },
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    recompute_callback = RecomputeBatchesCallback(
        train_dataset, val_dataset, cfg.data.data_sampling, train_ep_ids, val_ep_ids
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, recompute_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()