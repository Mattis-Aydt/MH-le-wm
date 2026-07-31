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
        steps_per_chunk=1,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_embedder = action_encoder
        self.action_chunk_encoder = action_chunk_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.steps_per_chunk = steps_per_chunk

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

    def _encode_chunks(self, act, chunk_len):
        """Encode (N, T, action_dim) actions grouped into length-`chunk_len` chunks.

        chunk_len=1 recovers unit chunks (each action its own chunk).
        Returns chunk latents (N, T // chunk_len, emb_dim).
        """
        N, T, _ = act.shape
        assert T % chunk_len == 0, f"T={T} not divisible by chunk_len={chunk_len}"
        a = self.action_embedder(act)  # (N, T, emb_dim)
        a = a.reshape(N * (T // chunk_len), chunk_len, -1)
        lengths = torch.full(
            (N * (T // chunk_len),), chunk_len, dtype=torch.long, device=act.device
        )
        c = self.action_chunk_encoder(a, lengths=lengths)  # (N*C, 1, emb_dim)
        return c.view(N, T // chunk_len, -1)

    def rollout(self, info, action_sequence, history_size: int = 3, steps_per_chunk: int = None):
        """Rollout the model given an initial info dict and action sequence.

        pixels: (B, 1, H, C, H, W) — H history frames as provided by the
            policy (H may be smaller than history_size; the window grows)
        action_sequence: (B, S, T, action_dim) — S CEM plan samples,
            first H entries are history actions, rest are future actions.

        steps_per_chunk (g): prediction jump per rollout step, in bundled
            steps. g=1 recovers the gap=1 regime exactly; g=2 matches
            models trained with fixed_gap=2.

        Mechanics (single code path for all g):
        - j0 = (H-1) % g: index of the oldest history frame that sits at
          spacing g from the current frame. History is subsampled to
          [j0, j0+g, ..., H-1] so context spacing matches training.
        - actions from index j0 on are encoded ONCE as length-g chunks,
          chunk-aligned with the subsampled frames (chunk j leaves obs j).
        - the loop only calls predict() autoregressively; window-relative
          time ids advance by g per step -> horizon_embed(g).
        - encode() is called ONCE for the history frames.
        """
        g = steps_per_chunk if steps_per_chunk is not None else self.steps_per_chunk

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)  # number of history frames provided
        B, S, T = action_sequence.shape[:3]

        j0 = (H - 1) % g  # oldest history frame index at spacing g from current
        n_act = T - j0    # actions from j0 onward (chunk-aligned)
        assert n_act % g == 0, (
            f"(T - j0)={n_act} not divisible by steps_per_chunk={g} "
            f"(H={H}, T={T}) — adjust plan horizon"
        )
        C = n_act // g                    # total chunks (history + future)
        M = (H - 1 - j0) // g + 1         # kept history frames
        assert C >= M, f"not enough actions (C={C}) for history (M={M})"

        # ---- one-time encode of history frames, subsampled to spacing g ----
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init.pop("action", None)  # actions are encoded separately below
        _init["pixels"] = _init["pixels"][:, j0::g]  # (B, M, ...)
        _init = self.encode(_init)  # emb: (B, M, D)
        emb = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()  # (B*S, M, D)

        # ---- one-time encode of ALL actions from j0 on, as length-g chunks ----
        act_all = rearrange(action_sequence, "b s ... -> (b s) ...")  # (B*S, T, adim)
        act_chunks = self._encode_chunks(act_all[:, j0:], g)  # (B*S, C, D)

        HS = history_size
        device = emb.device

        def _predict_next(emb, act_chunks):
            """One autoregressive step with a growing window.

            Window length k = min(HS, current obs count). Window-relative
            time ids [0, g, ..., k*g] — all spacings and the prediction
            horizon equal g, matching the fixed-gap training regime.
            """
            k = min(HS, emb.size(1))
            emb_trunc = emb[:, -k:]
            act_trunc = act_chunks[:, emb.size(1) - k : emb.size(1)]
            ids = torch.arange(0, k + 1, device=device) * g
            ids = ids.unsqueeze(0).expand(emb.size(0), -1)  # (B*S, k+1)
            pred = self.predict(emb_trunc, act_trunc, time_ids=ids)[:, -1:]
            return torch.cat([emb, pred], dim=1)

        # ---- autoregressive rollout: predict only ----
        n_steps = C - M  # future predictions beyond the final one
        for _ in range(n_steps):
            emb = _predict_next(emb, act_chunks)

        # predict the final state
        emb = _predict_next(emb, act_chunks)

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
        info_dict = self.rollout(info_dict, action_candidates, steps_per_chunk=self.steps_per_chunk)

        cost = self.criterion(info_dict)
        return cost