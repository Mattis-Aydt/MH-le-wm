import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from dataset import VariableHorizonDataset, variable_horizon_collate
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback

import torch.nn.functional as F


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]  # (B, T-1, D)

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

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]

    # time_ids shape: (B, T) where T = num_steps
    # predictor needs T+1 times for T positions, so slice ctx_len+1
    time_ids = batch.get("time_ids")
    if time_ids is not None:
        pred_time_ids = time_ids[:, : ctx_len + 1]
        pred_time_ids = pred_time_ids - pred_time_ids[:, :1]  # window-relative: first context = 0
    else:
        pred_time_ids = None

    pred_emb = self.model.predict(ctx_emb, ctx_act, time_ids=pred_time_ids)

    # per-sample prediction loss (B,) — training loss is still the plain mean
    per_sample = (pred_emb - tgt_emb).pow(2).mean(dim=(1, 2))
    output["pred_loss"] = per_sample.mean()

    # === horizon-split logging (LOGGING ONLY — not used for training) ===
    if time_ids is not None:
        gaps = time_ids[:, 1:] - time_ids[:, :-1]  # (B, T-1) absolute gaps
        short = (gaps == 1).all(dim=1)
        if short.any():
            self.log(f"{stage}/pred_loss_gap1", per_sample[short].mean().detach(), on_step=True)
        if (~short).any():
            self.log(f"{stage}/pred_loss_multistep", per_sample[~short].mean().detach(), on_step=True)

    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
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

    from stable_worldmodel.data.utils import get_cache_dir, _resolve_dataset
    datasets_dir = get_cache_dir(cache_dir, sub_folder='datasets')
    lance_path = os.path.abspath(_resolve_dataset(dataset_name, datasets_dir))

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    # Create dataset (transform=None for now, set after normalizers)
    dataset = VariableHorizonDataset(
        lance_path=lance_path,
        num_steps=cfg.data.dataset.num_steps,
        windows_per_episode_factor=cfg.data.get("windows_per_episode_factor", 1.0),
        max_gap=cfg.data.get("max_gap", 50),
        geometric_p=cfg.data.get("geometric_p", 0.5),
        buffer_size=cfg.data.get("buffer_size", 2000),
        frameskip=cfg.data.dataset.frameskip,
    )

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
        cfg.model.action_encoder.input_dim = dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    dataset._col_cache.clear()

    # Split by episode indices instead of random_split
    total_episodes = dataset.num_episodes
    n_train = int(total_episodes * cfg.train_split)

    train_dataset = VariableHorizonDataset(
        lance_path=lance_path,
        num_steps=cfg.data.dataset.num_steps,
        windows_per_episode_factor=cfg.data.get("windows_per_episode_factor", 1.0),
        max_gap=cfg.data.get("max_gap", 50),
        geometric_p=cfg.data.get("geometric_p", 0.5),
        buffer_size=cfg.data.get("buffer_size", 2000),
        frameskip=cfg.data.dataset.frameskip,
        transform=transform,
    )
    train_dataset.episode_order = list(range(n_train))

    val_dataset = VariableHorizonDataset(
        lance_path=lance_path,
        num_steps=cfg.data.dataset.num_steps,
        windows_per_episode_factor=cfg.data.get("val_windows_per_episode_factor", 0.2),
        max_gap=cfg.data.get("max_gap", 50),
        geometric_p=cfg.data.get("geometric_p", 0.5),
        buffer_size=cfg.data.get("buffer_size", 2000),
        frameskip=cfg.data.dataset.frameskip,
        transform=transform,
    )
    val_dataset.episode_order = list(range(n_train, total_episodes))

    train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=variable_horizon_collate,
        persistent_workers=False,
        prefetch_factor=cfg.loader.get("prefetch_factor", 3) if cfg.num_workers > 0 else None,
        pin_memory=cfg.loader.get("pin_memory", True),
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
    )
    val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=variable_horizon_collate,
        persistent_workers=False,
        prefetch_factor=cfg.loader.get("prefetch_factor", 3) if cfg.num_workers > 0 else None,
        pin_memory=cfg.loader.get("pin_memory", True),
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

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

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
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