"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        action_chunk_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_embedder = action_encoder
        self.action_chunk_encoder = action_chunk_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings."""

        pixels = info["pixels"].float()
        b, t = pixels.size(0), pixels.size(1)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b, t=t)

        if "action" in info:
            # info["action"]: (B, T, L_max, action_dim)
            # info["action_lengths"]: (B, T)
            actions = info["action"]
            lengths = info["action_lengths"]

            B, T, L, D = actions.shape
            # Flatten batch+time for embedder
            actions_flat = rearrange(actions, "b t l d -> (b t) l d")
            lengths_flat = rearrange(lengths, "b t -> (b t)")

            # Per-step embedding: (B*T, L, D) -> (B*T, L, emb_dim)
            act_emb_flat = self.action_embedder(actions_flat)

            # Chunk encoding with padding mask
            chunk_latents = self.action_chunk_encoder(act_emb_flat, lengths=lengths_flat)
            # (B*T, 1, emb_dim) -> (B, T, emb_dim)
            chunk_latents = rearrange(chunk_latents, "(b t) 1 d -> b t d", b=B, t=T)

            info["act_emb"] = chunk_latents

        return info

    def predict(self, emb, act_emb, time_ids=None):
        """Predict next state embedding."""
        preds = self.predictor(emb, act_emb, time_ids=time_ids)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def _encode_unit_chunks(self, act):
        """Encode (N, T, action_dim) bundled actions as length-1 chunks.

        Every action is an independent batch element for the chunk encoder
        (never merged across chunks). Eval uses fixed gap=1, so each bundled
        action IS a complete chunk — matching the gap=1 training regime.
        Returns chunk latents (N, T, emb_dim).
        """
        N, T, _ = act.shape
        a = self.action_embedder(act)  # (N, T, emb_dim)
        a = a.reshape(N * T, 1, -1)  # each action is its own length-1 chunk
        lengths = torch.ones(N * T, dtype=torch.long, device=act.device)
        c = self.action_chunk_encoder(a, lengths=lengths)  # (N*T, 1, emb_dim)
        return c.view(N, T, -1)

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.

        pixels: (B, 1, H, C, H, W) — H history frames as provided by the
            policy (H may be smaller than history_size; the window grows)
        action_sequence: (B, S, T, action_dim) — S CEM plan samples,
            first H entries are history actions, rest are future actions.

        Fixed-horizon (gap=1) rollout:
        - encode() ONCE for the history frames
        - all actions chunk-encoded ONCE upfront (unit chunks)
        - the loop only calls predict() autoregressively
        - time_ids are consecutive so every horizon = 1 (gap=1 regime)
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)  # number of history frames (= history_size)
        B, S, T = action_sequence.shape[:3]
        n_steps = T - H

        # ---- one-time encode of history frames ----
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init.pop("action", None)  # actions are encoded separately below
        _init = self.encode(_init)  # emb: (B, H, D)
        emb = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()  # (B*S, H, D)

        # ---- one-time encode of ALL actions (history + future) ----
        act_all = rearrange(action_sequence, "b s ... -> (b s) ...")  # (B*S, T, adim)
        act_emb_all = self._encode_unit_chunks(act_all)  # (B*S, T, D)

        HS = history_size
        device = emb.device

        def _predict_next(emb, act_emb_all):
            """One autoregressive step with a growing window.

            Window length k = min(HS, current obs count) — mirrors the
            original rollout (the policy provides only H=2 history frames,
            so the window grows 2 -> 3). Window-relative time ids [0..k]
            with all horizons = 1 (gap=1 regime), matching training.
            """
            k = min(HS, emb.size(1))
            emb_trunc = emb[:, -k:]
            act_trunc = act_emb_all[:, emb.size(1) - k : emb.size(1)]
            ids = torch.arange(0, k + 1, device=device)
            ids = ids.unsqueeze(0).expand(emb.size(0), -1)  # (B*S, k+1)
            pred = self.predict(emb_trunc, act_trunc, time_ids=ids)[:, -1:]
            return torch.cat([emb, pred], dim=1)

        # ---- autoregressive rollout: predict only ----
        for t in range(n_steps):
            emb = _predict_next(emb, act_emb_all)

        # predict the final state
        emb = _predict_next(emb, act_emb_all)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        return cost