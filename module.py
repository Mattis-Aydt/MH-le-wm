import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with optional causal masking and custom mask"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True, mask=None):
        """
        x    : (B, T, D)
        mask : optional, broadcastable to (B, heads, T, T)
               BoolTensor where True = attend, False = mask out
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)

        if mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=drop, is_causal=False
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=causal
            )

        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, mask=None, causal=True):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            causal=causal,
            mask=mask,
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, mask=None, causal=True):
        x = x + self.attn(self.norm1(x), causal=causal, mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None, mask=None, causal=True):
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            if isinstance(block, Block):
                x = block(x, mask=mask, causal=causal)
            else:
                x = block(x, c, mask=mask, causal=causal)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class ActionChunkEncoder(nn.Module):
    """Encodes a variable-length sequence of per-step action embeddings
    into a single action chunk latent with explicit horizon embedding.

    Uses bidirectional attention with padding masking — no causal mask needed.
    """

    def __init__(
        self,
        emb_dim,
        depth=2,
        heads=4,
        mlp_dim=512,
        dim_head=64,
        dropout=0.0,
        max_horizon=50,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_horizon = max_horizon

        # Horizon embedding table
        self.horizon_embed = nn.Embedding(max_horizon + 1, emb_dim)

        # Bidirectional Transformer over per-step action embeddings
        self.transformer = Transformer(
            input_dim=emb_dim,
            hidden_dim=emb_dim,
            output_dim=emb_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=Block,
        )

        self.output_norm = nn.LayerNorm(emb_dim)

    def forward(self, act_emb, lengths=None):
        """
        act_emb : (B, L, emb_dim) — per-step action embeddings (padded)
        lengths : (B,) — true length of each sequence before padding

        Returns : (B, 1, emb_dim) — single chunk latent per sample
        """
        B, L, D = act_emb.shape

        # Build padding mask: True = real token, False = padded token
        if lengths is not None:
            mask = (
                torch.arange(L, device=act_emb.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        else:
            mask = None

        # Bidirectional transformer (causal=False)
        chunk_latent = self.transformer(
            act_emb, mask=mask, causal=False
        )  # (B, L, emb_dim)

        # Take the LAST REAL token as the chunk summary
        if lengths is not None:
            last_idx = (lengths - 1).clamp(min=0)  # (B,)
            chunk_latent = chunk_latent[
                torch.arange(B, device=act_emb.device), last_idx
            ]  # (B, emb_dim)
            chunk_latent = chunk_latent.unsqueeze(1)  # (B, 1, emb_dim)
        else:
            chunk_latent = chunk_latent[:, -1, :]  # (B, emb_dim)
            chunk_latent = chunk_latent.unsqueeze(1)

        chunk_latent = self.output_norm(chunk_latent)

        # Add horizon embedding based on TRUE lengths
        if lengths is not None:
            h_embed = self.horizon_embed(
                lengths.clamp(max=self.max_horizon)
            )  # (B, emb_dim)
        else:
            h = torch.tensor(min(L, self.max_horizon), device=act_emb.device)
            h_embed = self.horizon_embed(h).unsqueeze(0).expand(B, -1)

        h_embed = h_embed.unsqueeze(1)  # (B, 1, emb_dim)
        chunk_latent = chunk_latent + h_embed

        return chunk_latent


class ARPredictor(nn.Module):
    """Autoregressive predictor with learned time and horizon embeddings."""

    def __init__(
        self,
        *,
        num_frames,
        max_time=500,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.max_time = max_time
        # Learned time embedding: indexed by actual cumulative time
        self.time_embed = nn.Embedding(max_time, input_dim)
        # Learned horizon embedding: indexed by chunk length
        self.horizon_embed = nn.Embedding(max_time, input_dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c, time_ids=None):
        """
        x: (B, T, d) — observation embeddings for T input positions
        c: (B, T, d) — action chunk latents for T chunks
        time_ids: (B, T+1) — cumulative times for T+1 observations
                  e.g. [0, 5, 8, 12] -> obs times for inputs + last target time
        """
        if time_ids is not None:
            # Time embeddings for the T input observations (all but last time)
            x = x + self.time_embed(time_ids[:, :-1])

            # Horizon = time difference between consecutive observations
            horizons = time_ids[:, 1:] - time_ids[:, :-1]  # (B, T)
            c = c + self.horizon_embed(horizons)

        x = self.dropout(x)
        x = self.transformer(x, c)
        return x