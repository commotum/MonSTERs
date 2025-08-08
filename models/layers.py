from typing import Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func  # type: ignore[import]
except ImportError:
    # Fallback to FlashAttention 2
    from flash_attn import flash_attn_func  # type: ignore[import]

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


# ===============================================================
# MonSTER (fast-scalar triad) positional embedding
# ===============================================================


class MonsterEmbedding(nn.Module):
    """Precompute scalar tables for MonSTER positional encodings."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        top_delta: int = 1024,
        skip_prefix: bool = False,
        prefix_len: int = 0,
        use_xy: bool = False,
        grid_w: int = 30,
        device=None,
    ):
        super().__init__()

        self.head_dim = int(head_dim)
        self.main_dim = (self.head_dim // 12) * 12
        self.num_freq = self.main_dim // 12
        self.max_pos = int(max_position_embeddings)
        self.base = float(base)
        self.unit = 1.0 / float(top_delta)
        self.skip_prefix = bool(skip_prefix)
        self.prefix_len = int(prefix_len)
        self.use_xy = bool(use_xy)
        self.grid_w = int(grid_w)

        if self.num_freq == 0:
            self.ch = nn.Buffer(torch.empty(0, 0), persistent=False)
            self.sh = nn.Buffer(torch.empty(0, 0), persistent=False)
            self.cx = nn.Buffer(torch.empty(0, 0), persistent=False)
            self.sx = nn.Buffer(torch.empty(0, 0), persistent=False)
            self.cy = nn.Buffer(torch.empty(0, 0), persistent=False)
            self.sy = nn.Buffer(torch.empty(0, 0), persistent=False)
            self.cz = nn.Buffer(torch.empty(0, 0), persistent=False)
            self.sz = nn.Buffer(torch.empty(0, 0), persistent=False)
            return

        j = torch.arange(self.num_freq, dtype=torch.float32, device=device)
        inv_freq = self.base ** (-j / self.num_freq)

        t = torch.arange(self.max_pos, dtype=torch.float32, device=device)
        if self.use_xy:
            idx = torch.clamp(t - self.prefix_len, min=0)
            y = (idx // self.grid_w).to(torch.float32)
            x = (idx % self.grid_w).to(torch.float32)
            z = torch.zeros_like(t)
        else:
            x = torch.zeros_like(t)
            y = torch.zeros_like(t)
            z = torch.zeros_like(t)

        phi = (t * self.unit).unsqueeze(-1) * inv_freq
        thx = (x * self.unit).unsqueeze(-1) * inv_freq
        thy = (y * self.unit).unsqueeze(-1) * inv_freq
        thz = (z * self.unit).unsqueeze(-1) * inv_freq

        ch = torch.cosh(phi)
        sh = torch.sinh(phi)
        cx = torch.cos(thx)
        sx = torch.sin(thx)
        cy = torch.cos(thy)
        sy = torch.sin(thy)
        cz = torch.cos(thz)
        sz = torch.sin(thz)

        if self.skip_prefix and self.prefix_len > 0:
            k = min(self.prefix_len, self.max_pos)
            ch[:k] = 1.0
            sh[:k] = 0.0
            cx[:k] = 1.0
            sx[:k] = 0.0
            cy[:k] = 1.0
            sy[:k] = 0.0
            cz[:k] = 1.0
            sz[:k] = 0.0

        self.ch = nn.Buffer(ch, persistent=False)
        self.sh = nn.Buffer(sh, persistent=False)
        self.cx = nn.Buffer(cx, persistent=False)
        self.sx = nn.Buffer(sx, persistent=False)
        self.cy = nn.Buffer(cy, persistent=False)
        self.sy = nn.Buffer(sy, persistent=False)
        self.cz = nn.Buffer(cz, persistent=False)
        self.sz = nn.Buffer(sz, persistent=False)

    def forward(self) -> Dict[str, torch.Tensor]:
        if self.num_freq == 0:
            return {"kind": "monster", "num_freq": 0}
        return {
            "kind": "monster",
            "ch": self.ch,
            "sh": self.sh,
            "cx": self.cx,
            "sx": self.sx,
            "cy": self.cy,
            "sy": self.sy,
            "cz": self.cz,
            "sz": self.sz,
        }


def apply_monster_pos_emb(
    q: torch.Tensor, k: torch.Tensor, tables: Dict[str, torch.Tensor], head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tables.get("num_freq", None) == 0:
        return q, k

    ch = tables["ch"]
    sh = tables["sh"]
    cx = tables["cx"]
    sx = tables["sx"]
    cy = tables["cy"]
    sy = tables["sy"]
    cz = tables["cz"]
    sz = tables["sz"]

    orig_dtype = q.dtype
    T, F = ch.shape
    main_dim = (head_dim // 12) * 12
    if main_dim == 0:
        return q, k

    def b(x):
        return x.unsqueeze(0).unsqueeze(2)

    ch_b, sh_b = b(ch), b(sh)
    cx_b, sx_b = b(cx), b(sx)
    cy_b, sy_b = b(cy), b(sy)
    cz_b, sz_b = b(cz), b(sz)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        x = x.to(ch.dtype)
        bs, T2, H, D = x.shape
        assert T2 == T
        main = x[..., :main_dim].view(bs, T, H, main_dim // 12, 12)
        tail = x[..., main_dim:]

        X = main[..., 0:4]
        Y = main[..., 4:8]
        Z = main[..., 8:12]

        t, x1, y, z = X.unbind(dim=-1)
        t1 = ch_b * t - sh_b * x1
        x2 = -sh_b * t + ch_b * x1
        y2 = cx_b * y - sx_b * z
        z2 = sx_b * y + cx_b * z
        X_out = torch.stack([t1, x2, y2, z2], dim=-1)

        t, x1, y, z = Y.unbind(dim=-1)
        t1 = ch_b * t - sh_b * y
        y2 = -sh_b * t + ch_b * y
        x2 = cy_b * x1 - sy_b * z
        z2 = sy_b * x1 + cy_b * z
        Y_out = torch.stack([t1, x2, y2, z2], dim=-1)

        t, x1, y, z = Z.unbind(dim=-1)
        t1 = ch_b * t - sh_b * z
        z2 = -sh_b * t + ch_b * z
        x2 = cz_b * x1 - sz_b * y
        y2 = sz_b * x1 + cz_b * y
        Z_out = torch.stack([t1, x2, y2, z2], dim=-1)

        main_out = torch.cat([X_out, Y_out, Z_out], dim=-1).reshape(bs, T, H, main_dim)
        if tail.numel() == 0:
            return main_out.to(orig_dtype)
        return torch.cat([main_out, tail], dim=-1).to(orig_dtype)

    return _apply(q), _apply(k)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # Positional encoding: RoPE or MonSTER
        if cos_sin is not None:
            if isinstance(cos_sin, tuple):
                cos, sin = cos_sin
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
            elif isinstance(cos_sin, dict) and cos_sin.get("kind") == "monster":
                query, key = apply_monster_pos_emb(query, key, cos_sin, self.head_dim)

        # flash attn
        attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
