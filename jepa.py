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
        state_transformer,
        projector=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.state_transformer = state_transformer
        self.projector = projector or nn.Identity()

    def encode(self, info):
        pixels = info['pixels'].float()
        b = pixels.size(0)
        if pixels.ndim == 4:  # (B, C, H, W) — single frame, add time dim
            pixels = pixels.unsqueeze(1)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def encode_state(self, info):
        emb_windows = info["emb"]  # (B, H, D)
        B, H, D = emb_windows.shape
        
        x = emb_windows + self.state_transformer.pos_embedding[:, :H]
        x = self.state_transformer.dropout(x)
        x = self.state_transformer.transformer(x)
        
        info["state"] = x[:, -1, :]  # (B, D)
        return info

    def predict(self, state_emb, act_emb):
        """Predict next state embedding from current state and action.
        
        Args:
            state_emb: (B, D) -- single state embeddings
            act_emb: (B, A_emb) -- single action embeddings
            
        Returns:
            preds: (B, D) -- predicted next state embeddings
        """
        x = torch.cat([state_emb, act_emb], dim=-1)  # (B, D + A_emb)
        preds = self.predictor(x)                   # (B, D)
        return preds

     ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        _, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v) and k != "action"}
        _init = self.encode(_init)
        _init = self.encode_state(_init)
        state = _init["state"].unsqueeze(1).expand(B, S, -1)  # (B, S, D)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        state = rearrange(state, "b s d -> (b s) d").clone()
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        pred_states = [state.unsqueeze(1)]  # (list of (B*S, 1, D))
        for t in range(n_steps):
            action = act_future[:, t : t + 1]       # (B*S, 1, 10)
            action_emb = self.action_encoder(action) # (B*S, 1, A_emb)
            act_emb = action_emb.squeeze(1)         # (B*S, A_emb)
            
            old_state = state.clone()  # SAVE BEFORE PREDICT
            state = self.predict(state, act_emb)    # (B*S, D)
            pred_states.append(state.unsqueeze(1))  # (B*S, 1, D)
            

        pred_states = torch.cat(pred_states, dim=1)   # (B*S, n_steps+1, D)


        # unflatten batch and sample dimensions
        pred_rollout = rearrange(pred_states, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_states"] = pred_rollout

        return info
    


    def criterion(self, info_dict: dict):
        """Compute the cost between predicted terminal state and goal state."""
        pred_states = info_dict["predicted_states"]  # (B, S, T-H+1, D)
        goal_state = info_dict["goal_state"]  # (B, D)

        # Take the last predicted state (terminal state)
        pred_terminal = pred_states[:, :, -1, :]  # (B, S, D)

        # Expand goal_state to match action samples dimension
        if goal_state.ndim == 2:
            goal_state = goal_state.unsqueeze(1)  # (B, 1, D)

        # MSE cost per action candidate
        cost = F.mse_loss(
            pred_terminal,
            goal_state.detach(),
            reduction="none",
        ).sum(dim=-1)  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):


        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # --- Goal encoding ---
        if "goal_pixels_window" in info_dict:
            goal = {"pixels": info_dict["goal_pixels_window"]}
            goal = self.encode(goal)
            goal_emb = goal["emb"]
        else:
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]
            for k in info_dict:
                if k.startswith("goal_"):
                    goal[k[len("goal_"):]] = goal.pop(k)
            goal.pop("action", None)
            goal = self.encode(goal)
            goal_emb = goal["emb"]  # (B, 1, D)

        info_dict["goal_state"] = self.encode_state({"emb": goal_emb})["state"]
    

        # --- Rollout ---
        info_dict = self.rollout(info_dict, action_candidates)
        cost = self.criterion(info_dict)

        print(f"[get_cost] costs: min={cost.min().item():.4f}, mean={cost.mean().item():.4f}, max={cost.max().item():.4f}, std={cost.std().item():.4f}")




        return cost