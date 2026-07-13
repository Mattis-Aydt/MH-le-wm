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

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback



def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]          # (B, T, D)
    act_emb = output["act_emb"]  # (B, T, A_emb)
    B, T, D = emb.shape

    # ------------------------------------------------------------------
    # Build sliding windows of length ctx_len over the frame embeddings
    # windows: (B, T - ctx_len + 1, ctx_len, D)
    # ------------------------------------------------------------------
    windows = emb.unfold(dimension=1, size=ctx_len, step=1)
    windows = windows.permute(0, 1, 3, 2)  # -> (B, n_windows, ctx_len, D)

    T_valid = windows.size(1) - n_preds
    if T_valid <= 0:
        output["pred_loss"] = torch.tensor(0.0, device=emb.device)
        output["sigreg_loss"] = torch.tensor(0.0, device=emb.device)
        output["loss"] = torch.tensor(0.0, device=emb.device)
        losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
        return output

    # Input windows
    state_windows = windows[:, :T_valid]  # (B, T_valid, 3, D)
    flat_states = state_windows.reshape(B * T_valid, ctx_len, D)
    states = self.model.encode_state({"emb": flat_states})["state"]       # (B*T_valid, D)

    # TARGET: single frame, no overlap
    target_frames = emb[:, ctx_len + n_preds - 1:]  # (B, T_valid, D)
    flat_targets = target_frames.unsqueeze(2).reshape(B * T_valid, 1, D)
    next_states = self.model.encode_state({"emb": flat_targets})["state"]  # (B*T_valid, D)

    # === RESHAPE BACK ===
    states = states.reshape(B, T_valid, D)
    next_states = next_states.reshape(B, T_valid, D)

    # Actions at the end of each input window
    actions = act_emb[:, ctx_len - 1 : T - n_preds]  # (B, T_valid, A_emb)

    # Predict next state embeddings from current state + action
    flat_states = states.reshape(B * T_valid, D)
    flat_actions = actions.reshape(B * T_valid, -1)
    pred_next_states = self.model.predict(flat_states, flat_actions)
    pred_next_states = pred_next_states.reshape(B, T_valid, D)

    # LeWM loss
    output["pred_loss"] = (pred_next_states - next_states).pow(2).mean()
    
    # SIGReg on the *state* embeddings, not raw frame embeddings
    all_states = torch.cat([states, next_states], dim=0)  # (2*B, T_valid, D)
    output["sigreg_loss"] = self.sigreg(all_states.transpose(0, 1))
    
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
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # MANUAL CHECKPOINT LOADING
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)
    
    # Find latest checkpoint in run_dir or subfolders
    ckpt_files = []
    for path in [run_dir] + list(run_dir.iterdir()):
        if path.is_dir():
            ckpt_files.extend(path.glob("*.pt"))
            ckpt_files.extend(path.glob("*.ckpt"))
    
    if ckpt_files:
        ckpt_path = sorted(ckpt_files)[-1]  # latest
        print(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        world_model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully")
    else:
        print("No checkpoint found, starting from scratch")

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