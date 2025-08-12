Task 1 – Code Extraction
A. Positional Encoding Methods
Rotary Encoding

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        …
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

MonSTER Encoding

class MonsterEmbedding(nn.Module):
    …
    pos = torch.arange(self.max_pos, dtype=torch.float32, device=device)
    if self.use_xy:
        idx = torch.clamp(pos - self.prefix_len, min=0)
        y = (idx // self.grid_w).to(torch.float32)
        x = (idx % self.grid_w).to(torch.float32)
        sub_grid = int(self.grid_w ** 0.5)
        z = torch.floor(x / sub_grid) + torch.floor(y / sub_grid) * sub_grid
    else:
        x = torch.zeros_like(pos)
        y = torch.zeros_like(pos)
        z = torch.zeros_like(pos)
    …
    self.ch = nn.Buffer(ch, persistent=False)
    self.sh = nn.Buffer(sh, persistent=False)
    self.cx = nn.Buffer(cx, persistent=False)
    self.sx = nn.Buffer(sx, persistent=False)
    self.cy = nn.Buffer(cy, persistent=False)
    self.sy = nn.Buffer(sy, persistent=False)
    self.cz = nn.Buffer(cz, persistent=False)
    self.sz = nn.Buffer(sz, persistent=False)

def apply_monster_pos_emb(
    q: torch.Tensor, k: torch.Tensor, tables: Dict[str, torch.Tensor], head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tables.get("num_freq", None) == 0:
        return q, k
    …
    return _apply(q), _apply(k)

Learned Positional Embedding

class CastedEmbedding(nn.Module):
    …
    self.embedding_weight = nn.Parameter(
        trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
    )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))

elif self.config.pos_encodings == "learned":
    self.embed_pos = CastedEmbedding(
        self.config.seq_len + self.puzzle_emb_len,
        self.config.hidden_size,
        init_std=embed_init_std,
        cast_to=self.forward_dtype,
    )

Integration and Selection Logic

# HierarchicalReasoningModel_ACTV1_Inner.__init__
if self.config.pos_encodings == "rope":
    self.rotary_emb = RotaryEmbedding(...)
elif self.config.pos_encodings == "learned":
    self.embed_pos = CastedEmbedding(...)
elif self.config.pos_encodings == "monster":
    self.monster_emb = MonsterEmbedding(...)
else:
    raise NotImplementedError()

# _input_embeddings: add learned positional embeddings
if self.config.pos_encodings == "learned":
    # scale by 1/sqrt(2) to maintain forward variance
    embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

# HierarchicalReasoningModel_ACTV1_Inner.forward
pos_obj = None
if hasattr(self, "rotary_emb"):
    pos_obj = self.rotary_emb()
    if (
        self.config.pos_encodings == "rope"
        and self.config.skip_prefix
        and self.puzzle_emb_len > 0
    ):
        cos, sin = pos_obj
        cos = cos.clone(); sin = sin.clone()
        k = int(self.puzzle_emb_len)
        cos[:k].fill_(1.0); sin[:k].zero_()
        pos_obj = (cos, sin)
elif hasattr(self, "monster_emb"):
    pos_obj = self.monster_emb()
seq_info = dict(cos_sin=pos_obj)

# Attention.forward
if cos_sin is not None:
    if isinstance(cos_sin, tuple):
        cos, sin = cos_sin
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
    elif isinstance(cos_sin, dict) and cos_sin.get("kind") == "monster":
        query, key = apply_monster_pos_emb(query, key, cos_sin, self.head_dim)
        key = key * self.minkowski_mask.to(key.dtype)

B. Dataloading
def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata

class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        …
        assert self.config.global_batch_size % self.config.num_replicas == 0, \
            f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

C. Dataset Creation & Spatial Mapping
Flattening 2D grid indices into 1D

idx = torch.clamp(pos - self.prefix_len, min=0)
y = (idx // self.grid_w).to(torch.float32)
x = (idx % self.grid_w).to(torch.float32)
sub_grid = int(self.grid_w ** 0.5)
z = torch.floor(x / sub_grid) + torch.floor(y / sub_grid) * sub_grid

Configuration Parameters

# MonSTER-related config
monster_theta: float = 10000.0
monster_top_delta: int = 9
monster_use_xy: bool = True
monster_grid_w: int = 9

# Prefix length derived from puzzle embedding dimensionality
self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
…
self.monster_emb = MonsterEmbedding(
    head_dim=self.config.hidden_size // self.config.num_heads,
    max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
    base=self.config.monster_theta,
    top_delta=self.config.monster_top_delta,
    skip_prefix=self.config.skip_prefix,
    prefix_len=self.puzzle_emb_len,
    use_xy=self.config.monster_use_xy,
    grid_w=self.config.monster_grid_w,
)

Task 2 – Analysis of Current Implementation
1. Spatial Data Handling
The model treats spatial inputs by “flattening then applying” positional encodings:

A 2‑D (or 3‑D) grid cell at global index pos is decomposed into coordinates using integer division and modulus (y = idx // grid_w, x = idx % grid_w) and an optional sub‑grid for a z‑index.

These coordinates are converted into angular offsets and used to build trigonometric and hyperbolic lookup tables (cos, sin, cosh, sinh), stored as buffers for later use.

During the forward pass, each attention head retrieves the precomputed tables and applies them through vectorized operations (apply_rotary_pos_emb or apply_monster_pos_emb), treating the flattened sequence like any other 1‑D transformer input.

2. Efficiency for Sequential Data
Compatibility: Flattening allows reuse of standard Transformer layers and data loaders that operate on [batch, seq_len, dim] tensors. The selection logic in HierarchicalReasoningModel_ACTV1_Inner simply chooses between learned, RoPE, or MonSTER embeddings, feeding a tuple or dict into the attention module without altering downstream APIs.

Pre‑computation: Both rotary and MonSTER encodings precompute their trigonometric tables for the maximum sequence length and cache them as buffers (e.g., self.cos_cached, self.sin_cached, self.ch, self.sh) so that runtime cost is constant per forward pass.

Vectorization: Application of embeddings relies on broadcasted tensor algebra (q * cos.unsqueeze(-2), rotate_half, batched hyperbolic rotations), enabling PyTorch to leverage highly optimized GPU kernels rather than Python loops.

3. Inherent Limitations
Flattening spatial grids into a linear sequence is a convenient hack but loses explicit neighbor relationships. The transformer only sees positional indices, so any non‑grid or irregular spatial structure must still be squeezed into the same 1‑D scheme. This approach becomes brittle for large or dynamic spatial layouts and makes it harder to encode complex adjacency information or support variable‑dimensional data. Consequently, while efficient for existing sequential tooling, it is not a scalable, long‑term solution for genuine multi‑dimensional spatial reasoning.

---

v2

Repository Deep-Dive Report – MonSTERs Positional Encoding & Data Pipeline
1. File Index (Paths + Why Relevant)
models/layers.py – Core positional encodings (RoPE/MonSTER), helpers, and attention wiring.

models/hrm/hrm_act_v1.py – Model configuration and integration points for positional encodings.

puzzle_dataset.py – Iterable dataset with batching, padding, and sequence-first memmap loading.

dataset/common.py – Dataset metadata schema and dihedral transforms for augmentation.

dataset/build_arc_dataset.py – Example builder converting 2D ARC puzzles into flattened token sequences.

dataset/build_sudoku_dataset.py – Builder showing sequence flattening and metadata creation for Sudoku.

pretrain.py – Training harness that constructs DataLoader, passes metadata/config, and compiles the model.

config/arch/hrm_v1.yaml – Default architecture config enabling pos_encodings selection.

MonSTER/note_free.py – Reference JAX implementation of MonSTER rotor generation.

2. Extracted Code (verbatim, grouped by topic)
Positional Encodings
models/layers.py

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
models/layers.py

class MonsterEmbedding(nn.Module):
    """Precompute scalar tables for MonSTER positional encodings."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        top_delta: int = 9,
        skip_prefix: bool = True,
        prefix_len: int = 0,
        use_xy: bool = True,
        grid_w: int = 9,
        device=None,
    ):
        super().__init__()

        self.head_dim = int(head_dim)
        self.main_dim = (self.head_dim // 12) * 12
        self.num_freq = self.main_dim // 12
        self.max_pos = int(max_position_embeddings)
        self.base = float(base)
        self.unit = torch.pi / float(top_delta)
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

        pos = torch.arange(self.max_pos, dtype=torch.float32, device=device)
        if self.use_xy:
            idx = torch.clamp(pos - self.prefix_len, min=0)
            y = (idx // self.grid_w).to(torch.float32)
            x = (idx % self.grid_w).to(torch.float32)
            sub_grid = int(self.grid_w ** 0.5)
            z = torch.floor(x / sub_grid) + torch.floor(y / sub_grid) * sub_grid
        else:
            x = torch.zeros_like(pos)
            y = torch.zeros_like(pos)
            z = torch.zeros_like(pos)

        t = torch.zeros_like(pos)

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
models/layers.py

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
models/layers.py

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
MonSTER/note_free.py

import jax
import jax.numpy as jnp

def get_monster_rotors(
    pos_q,
    pos_k,
    num_blocks: int,
    s: float = 1.0,
    c: float = 299792458.0,
    base_time: float = 10000.,
    base_space: float = 10000.,
    epsilon: float = 1e-8,
    dtype=jnp.float32
):

    pos_q = jnp.asarray(pos_q, dtype=dtype)
    pos_k = jnp.asarray(pos_k, dtype=dtype)

    tau = s / c

    delta_pos_raw = pos_k - pos_q
    delta_t_raw = delta_pos_raw[..., 0]
    delta_coords_raw = delta_pos_raw[..., 1:]

    delta_n_t = delta_t_raw / tau
    delta_n_coords = delta_coords_raw / s

    return _compute_rotors_from_normalized_displacements(
        delta_n_t=delta_n_t,
        delta_n_coords=delta_n_coords,
        num_blocks=num_blocks,
        base_time=base_time,
        base_space=base_space,
        epsilon=epsilon,
        dtype=dtype
    )
Attention Integration
models/layers.py

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

        pattern12 = torch.tensor([1, -1, -1, -1] * 3, dtype=torch.float32)
        main_dim = (self.head_dim // 12) * 12
        mask = torch.ones(self.head_dim, dtype=torch.float32)
        if main_dim > 0:
            F = main_dim // 12
            mask[:main_dim] = pattern12.repeat(F)
        mask = mask.view(1, 1, 1, self.head_dim)
        self.register_buffer("minkowski_mask", mask, persistent=False)

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
                key = key * self.minkowski_mask.to(key.dtype)

        # flash attn
        attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)
Dataset & Dataloading
dataset/common.py

class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int

    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int

    total_groups: int
    mean_puzzle_examples: float

    sets: List[str]
puzzle_dataset.py

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)
puzzle_dataset.py

class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool

    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.

    rank: int
    num_replicas: int
puzzle_dataset.py

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        with open(os.path.join(self.config.dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            # Load subset
            self._data[set_name] = {
                field_name: np.load(os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size

            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,

                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}
puzzle_dataset.py

    def _iter_test(self):
        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)

                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)

                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)

                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield set_name, batch, end_index - start_index

                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0

            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                yield set_name, batch, global_effective_batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."

        self._lazy_load_dataset()

        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
dataset/build_arc_dataset.py

def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    # Compute random top-left pad
    if do_translation:
        pad_r = np.random.randint(0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1)
        pad_c = np.random.randint(0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1)
    else:
        pad_r = pad_c = 0

    # Pad grid
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(grid + 2, ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)), constant_values=0)

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result
dataset/build_sudoku_dataset.py

def _seq_to_numpy(seq):
    arr = np.concatenate(seq).reshape(len(seq), -1)

    assert np.all((arr >= 0) & (arr <= 9))
    return arr + 1

results = {
    "inputs": _seq_to_numpy(results["inputs"]),
    "labels": _seq_to_numpy(results["labels"]),

    "group_indices": np.array(results["group_indices"], dtype=np.int32),
    "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
    "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
}
Training Harness Glue
pretrain.py

class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []
pretrain.py

def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,

        dataset_path=config.data_path,

        rank=rank,
        num_replicas=world_size,

        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata
pretrain.py

def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // world_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)
config/arch/hrm_v1.yaml

# options: rope, learned, monster
pos_encodings: rope
skip_prefix: true
models/hrm/hrm_act_v1.py

class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: Literal["learned", "rope", "monster"]

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # MonSTER params
    monster_theta: float = 10000.0
    monster_top_delta: int = 9
    monster_use_xy: bool = True
    monster_grid_w: int = 9

    # Don't apply position encoding to first k tokens if True
    skip_prefix: bool = True
models/hrm/hrm_act_v1.py

if self.config.pos_encodings == "rope":
    self.rotary_emb = RotaryEmbedding(
        dim=self.config.hidden_size // self.config.num_heads,
        max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
        base=self.config.rope_theta,
    )
elif self.config.pos_encodings == "learned":
    self.embed_pos = CastedEmbedding(
        self.config.seq_len + self.puzzle_emb_len,
        self.config.hidden_size,
        init_std=embed_init_std,
        cast_to=self.forward_dtype,
    )
elif self.config.pos_encodings == "monster":
    self.monster_emb = MonsterEmbedding(
        head_dim=self.config.hidden_size // self.config.num_heads,
        max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
        base=self.config.monster_theta,
        top_delta=self.config.monster_top_delta,
        skip_prefix=self.config.skip_prefix,
        prefix_len=self.puzzle_emb_len,
        use_xy=self.config.monster_use_xy,
        grid_w=self.config.monster_grid_w,
    )
else:
    raise NotImplementedError()
models/hrm/hrm_act_v1.py

def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
    # Token embedding
    embedding = self.embed_tokens(input.to(torch.int32))

    # Puzzle embeddings
    if self.config.puzzle_emb_ndim > 0:
        puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

        pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
        if pad_count > 0:
            puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

        embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

    # Position embeddings
    if self.config.pos_encodings == "learned":
        # scale by 1/sqrt(2) to maintain forward variance
        embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

    # Scale
    return self.embed_scale * embedding
models/hrm/hrm_act_v1.py

def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    pos_obj = None
    if hasattr(self, "rotary_emb"):
        pos_obj = self.rotary_emb()
        if (
            self.config.pos_encodings == "rope"
            and self.config.skip_prefix
            and self.puzzle_emb_len > 0
        ):
            cos, sin = pos_obj
            cos = cos.clone()
            sin = sin.clone()
            k = int(self.puzzle_emb_len)
            cos[:k].fill_(1.0)
            sin[:k].zero_()
            pos_obj = (cos, sin)
    elif hasattr(self, "monster_emb"):
        pos_obj = self.monster_emb()

    seq_info = dict(cos_sin=pos_obj)
3. Call Graphs & Data Shapes (succinct)
Data Pipeline & Shapes
PuzzleDataset.__iter__ (train/test):
  yields (set_name, batch, effective_batch_size)
    batch["inputs"]:        [B_local, seq_len]
    batch["labels"]:        [B_local, seq_len]
    batch["puzzle_identifiers"]: [B_local]
      (padding handled inside _collate_batch)

Pretrain.create_dataloader -> DataLoader(...)  # streaming IterableDataset
Training loop -> model.inner.forward(batch)

HierarchicalReasoningModel_ACTV1_Inner._input_embeddings:
  input (tokens) -> embed_tokens -> [B, seq_len, hidden_size]
  puzzle_emb (optional) -> expanded & prepended -> [B, seq_len + puzzle_emb_len, hidden_size]
  (learned pos adds embedding; RoPE/MonSTER handled later)
  returns scaled embedding [B, T_total, hidden_size]

Inside forward():
  pos_obj = RotaryEmbedding() -> (cos, sin) each [T_total, head_dim]
             or MonsterEmbedding() -> dict of tables [T_total, num_freq]
  seq_info = {"cos_sin": pos_obj}
  L_level / H_level modules apply Attention with seq_info.

Attention.forward:
  hidden_states -> qkv_proj -> [B, T_total, (H+2H)*D]
  reshape -> query/key/value: [B, T_total, num_heads, head_dim]
  apply_rotary_pos_emb or apply_monster_pos_emb (broadcast over seq_len/head)
  after Monster: key *= minkowski_mask [1,1,1,head_dim]
  flash_attn_func -> output [B, T_total, head_dim*num_heads]
Position Mapping (MonSTER Embedding)
pos = arange(max_pos)
idx = clamp(pos - prefix_len, 0)
y = idx // grid_w
x = idx % grid_w
sub_grid = sqrt(grid_w)
z = floor(x/sub_grid) + floor(y/sub_grid)*sub_grid
t = 0
Positions are encoded into (t, x, y, z) then scaled by unit and inv_freq to produce cosh/sinh/cos/sin tables.

Tail/Main Head Dimensions
head_dim decomposed: main_dim = floor(head_dim/12)*12

Frequencies: num_freq = main_dim/12

Q/K tensors split into 3 groups of 4 (X, Y, Z) × num_freq; any remainder (tail) bypasses encoding.

4. Efficiency Notes
Vectorized sinusoid caching: RotaryEmbedding and MonsterEmbedding precompute frequency tables once and store buffers, avoiding per-token trig calls

Skip-prefix handling: Prefix tokens are zeroed/ones in tables to bypass encoding without runtime conditionals

Broadcast-heavy apply: Both apply_rotary_pos_emb and apply_monster_pos_emb rely on broadcasting and reshaping rather than loops for token-wise rotation

Minkowski mask: Precomputed [1,-1,-1,-1]*F pattern multiplied once to enforce metric signature with minimal overhead

FlashAttention: Attention forward delegates to fused flash_attn_func for memory-efficient softmax-attention

Torch.compile: Model compiled for static graph execution when env var allows, reducing Python overhead

Sequence-first memmap: _lazy_load_dataset loads arrays via np.load(..., mmap_mode) to avoid copying and keep contiguous memory

Batch packing: _sample_batch fills batches by shuffling group/puzzle indices in NumPy, avoiding Python loops during iteration

Pinned-memory DataLoader: DataLoader uses prefetch_factor=8, pin_memory=True, and persistent_workers=True for efficient host-device transfers and worker reuse

5. Fragility & Coupling (sequence-centric assumptions)
Linear index → grid mapping baked into MonsterEmbedding: uses grid_w and assumes square sub-grid; only (x,y,z) derived from 1D index.

Head-dim divisibility: MonSTER requires head_dim % 12 == 0 for full coverage; tails bypass encoding but still assume a sequence layout.

Cos/Sin cache shape: positional buffers shaped [seq_len, head_dim]; any 2D/3D structure is flattened prior to encoding.

Dataset API: PuzzleDataset emits only flat token sequences and integer puzzle identifiers; no positional dictionary is propagated.

Builders flatten grids: ARC/Sudoku converters turn 2D grids into 1D token arrays with manual padding, losing explicit coordinate info.

Config & training path: Seq length threaded everywhere (PretrainConfig, model config, attention) and expected to be constant per batch.

6. Hook Points for Refactor
Dataset layer

PuzzleDataset.__iter__ & _collate_batch: return an additional pos_dict (e.g., {t,x,y,z}) alongside tokens.

Dataset builders (build_arc_dataset, build_sudoku_dataset) generate and store coordinate arrays or functions producing them.

DataLoader/Training harness

create_dataloader and training loop: expect batches containing pos_dict; preserve dtypes/devices.

Model input path

_input_embeddings and forward in HierarchicalReasoningModel_ACTV1_Inner: accept pos_dict; skip prefix logic driven by dictionary instead of index math.

Positional encoding registry

Wrap RotaryEmbedding, MonsterEmbedding, and future learned encoders behind a common interface (PositionalEncodingBase.forward(q,k,pos_dict)).

Attention

Attention.forward takes a unified pos_obj (from registry) with explicit head_dim parameter.

Caching strategy

Positional modules maintain per-device caches keyed by max sequence/grid extents; pos_dict shapes drive cache selection.

Config

Extend config files (arch.pos_encodings, monster_*, etc.) to drive registry selection and grid parameters without assuming linear indexing.

7. Minimal Plan (bullet list)
Dataset emission

Builders output (tokens, labels, pos_dict); pos_dict holds arrays for each spatial/temporal axis.

PuzzleDataset._collate_batch pads and converts each pos_dict entry to tensors, preserving shape [B_local, T].

Collate & Dataloader

create_dataloader unchanged except for collated batch structure; ensure pin_memory handles pos_dict tensors.

Model interface

HierarchicalReasoningModel_ACTV1_Inner.forward(batch, pos_dict) → pass pos_dict to positional-encoding factory.

Positional encoding registry

Factory: PosEncoding.get(name, **cfg) returning module with forward(q, k, pos_dict, head_dim).

Implement wrappers for existing RotaryEmbedding and MonsterEmbedding.

Attention module

Accept pos_module object and call its forward(q, k, pos_dict); Minkowski mask applied internally for MonSTER.

Caching

Each positional encoder caches per-(device, max_len) tables; pos_dict drives lookup or generation.

Config wiring

Extend ArchConfig / YAML to carry new grid parameters, enabling seamless toggle between learned | rope | monster.

Testing
No tests were executed; analysis based solely on static code inspection.

Notes
The repository currently assumes 1D sequence order throughout; multi-dimensional positional dictionaries will require careful API design to avoid breaking the existing fast path.


---


v3

Repo Recon Report — MonSTERs Positional Encodings & Sudoku Data Pipeline
1. File Index (path → relevance)
dataset/build_sudoku_dataset.py — pulls Sudoku puzzles from HF, applies digit-remap + band/stack shuffles + transpose, flattens to [N,81], and writes .npy + metadata

dataset/common.py — dihedral transforms (shared augmentations) and dataset metadata shape definitions

puzzle_dataset.py — memory-mapped .npy loader, group-wise batching, padding, and torch DataLoader integration

models/hrm/hrm_act_v1.py — pydantic config threads pos_encodings, monster_*, rope_theta, skip_prefix, prefix_len, use_xy, grid_w; constructs embeddings and passes pos caches into attention

config/arch/hrm_v1.yaml — Hydra/Pydantic arch defaults (pos_encodings, skip_prefix, heads, dims)

models/layers.py — rotary + MonSTER encoding kernels, Minkowski mask, FlashAttention wrapper

pretrain.py — creates DataLoader with pin_memory, persistent_workers, prefetch_factor, and compiles model with torch.compile

2. Extracted Code (verbatim)
Positional encodings
models/layers.py

def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)                      # [D]→[-x2,x1]
                                                             # ​:codex-file-citation[codex-file-citation]{line_range_start=23 line_range_end=27 path=models/layers.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/layers.py#L23-L27"}​
def apply_rotary_pos_emb(q, k, cos, sin):
    # q,k: [bs,seq_len,num_heads,head_dim]; cos,sin: [seq_len,head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype); k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)    # ​:codex-file-citation[codex-file-citation]{line_range_start=30 line_range_end=40 path=models/layers.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/layers.py#L30-L40"}​
models/layers.py

class MonsterEmbedding(nn.Module):
    """Precompute scalar tables for MonSTER positional encodings."""
    def __init__(self, head_dim, max_position_embeddings, base=10000.0,
                 top_delta=9, skip_prefix=True, prefix_len=0,
                 use_xy=True, grid_w=9, device=None):
        ...
        self.head_dim = int(head_dim)
        self.main_dim = (self.head_dim // 12) * 12            # triad multiple
        self.num_freq = self.main_dim // 12
        ...
        pos = torch.arange(self.max_pos, dtype=torch.float32, device=device)
        if self.use_xy:
            idx = torch.clamp(pos - self.prefix_len, min=0)
            y = (idx // self.grid_w).to(torch.float32)
            x = (idx % self.grid_w).to(torch.float32)
            sub_grid = int(self.grid_w ** 0.5)
            z = torch.floor(x / sub_grid) + torch.floor(y / sub_grid) * sub_grid
        else:
            x = torch.zeros_like(pos); y = torch.zeros_like(pos); z = torch.zeros_like(pos)
        ...
        ch = torch.cosh(phi); sh = torch.sinh(phi)
        cx = torch.cos(thx);  sx = torch.sin(thx)
        cy = torch.cos(thy);  sy = torch.sin(thy)
        cz = torch.cos(thz);  sz = torch.sin(thz)
        if self.skip_prefix and self.prefix_len > 0:
            k = min(self.prefix_len, self.max_pos)
            ch[:k] = 1.0; sh[:k] = 0.0; cx[:k] = 1.0; sx[:k] = 0.0
            cy[:k] = 1.0; sy[:k] = 0.0; cz[:k] = 1.0; sz[:k] = 0.0
        self.ch = nn.Buffer(ch, persistent=False); self.sh = nn.Buffer(sh, persistent=False)
        self.cx = nn.Buffer(cx, persistent=False); self.sx = nn.Buffer(sx, persistent=False)
        self.cy = nn.Buffer(cy, persistent=False); self.sy = nn.Buffer(sy, persistent=False)
        self.cz = nn.Buffer(cz, persistent=False); self.sz = nn.Buffer(sz, persistent=False)
    def forward(self):
        if self.num_freq == 0: return {"kind": "monster", "num_freq": 0}
        return {"kind": "monster", "ch": self.ch, "sh": self.sh,
                "cx": self.cx, "sx": self.sx, "cy": self.cy,
                "sy": self.sy, "cz": self.cz, "sz": self.sz}       # ​:codex-file-citation[codex-file-citation]{line_range_start=48 line_range_end=151 path=models/layers.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/layers.py#L48-L151"}​
models/layers.py

def apply_monster_pos_emb(q, k, tables, head_dim):
    if tables.get("num_freq", None) == 0: return q, k
    ch, sh = tables["ch"], tables["sh"]; cx, sx = tables["cx"], tables["sx"]
    cy, sy = tables["cy"], tables["sy"]; cz, sz = tables["cz"], tables["sz"]
    T, F = ch.shape
    main_dim = (head_dim // 12) * 12
    def b(x): return x.unsqueeze(0).unsqueeze(2)             # broadcast
    ch_b, sh_b = b(ch), b(sh); cx_b, sx_b = b(cx), b(sx)
    cy_b, sy_b = b(cy), b(sy); cz_b, sz_b = b(cz), b(sz)
    def _apply(x):
        x = x.to(ch.dtype); bs, T2, H, D = x.shape; assert T2 == T
        main = x[..., :main_dim].view(bs, T, H, main_dim // 12, 12)
        tail = x[..., main_dim:]
        X, Y, Z = main[..., 0:4], main[..., 4:8], main[..., 8:12]
        t,x1,y,z = X.unbind(-1); t1 = ch_b*t - sh_b*x1; x2 = -sh_b*t + ch_b*x1
        y2 = cx_b*y - sx_b*z; z2 = sx_b*y + cx_b*z
        X_out = torch.stack([t1, x2, y2, z2], dim=-1)
        ... (Y,Z blocks similar) ...
        main_out = torch.cat([X_out, Y_out, Z_out], dim=-1).reshape(bs,T,H,main_dim)
        return main_out.to(orig_dtype) if tail.numel()==0 else torch.cat([main_out,tail],dim=-1).to(orig_dtype)
    orig_dtype = q.dtype
    return _apply(q), _apply(k)                              # ​:codex-file-citation[codex-file-citation]{line_range_start=154 line_range_end=220 path=models/layers.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/layers.py#L154-L220"}​
models/layers.py

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)
    def forward(self):
        return self.cos_cached, self.sin_cached              # ​:codex-file-citation[codex-file-citation]{line_range_start=260 line_range_end=275 path=models/layers.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/layers.py#L260-L275"}​
Attention integration
models/layers.py

pattern12 = torch.tensor([1, -1, -1, -1] * 3, dtype=torch.float32)
main_dim = (self.head_dim // 12) * 12
mask = torch.ones(self.head_dim, dtype=torch.float32)
if main_dim > 0:
    F = main_dim // 12
    mask[:main_dim] = pattern12.repeat(F)
mask = mask.view(1, 1, 1, self.head_dim)
self.register_buffer("minkowski_mask", mask, persistent=False)   # ​:codex-file-citation[codex-file-citation]{line_range_start=292 line_range_end=299 path=models/layers.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/layers.py#L292-L299"}​

def forward(self, cos_sin, hidden_states):
    batch_size, seq_len, _ = hidden_states.shape
    qkv = self.qkv_proj(hidden_states).view(batch_size, seq_len,
         self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    query = qkv[:, :, :self.num_heads]
    key   = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
    value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
    if cos_sin is not None:
        if isinstance(cos_sin, tuple):
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        elif isinstance(cos_sin, dict) and cos_sin.get("kind") == "monster":
            query, key = apply_monster_pos_emb(query, key, cos_sin, self.head_dim)
            key = key * self.minkowski_mask.to(key.dtype)
    attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
    if isinstance(attn_output, tuple): attn_output = attn_output[0]
    attn_output = attn_output.view(batch_size, seq_len, self.output_size)
    return self.o_proj(attn_output)                              # ​:codex-file-citation[codex-file-citation]{line_range_start=301 line_range_end=329 path=models/layers.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/layers.py#L301-L329"}​
models/hrm/hrm_act_v1.py

class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int; seq_len: int; puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int; vocab_size: int
    H_cycles: int; L_cycles: int; H_layers: int; L_layers: int
    hidden_size: int; expansion: float; num_heads: int
    pos_encodings: Literal["learned","rope","monster"]
    rms_norm_eps: float = 1e-5; rope_theta: float = 10000.0
    monster_theta: float = 10000.0; monster_top_delta: int = 9
    monster_use_xy: bool = True; monster_grid_w: int = 9
    skip_prefix: bool = True; halt_max_steps: int = 16
    halt_exploration_prob: float; forward_dtype: str = "bfloat16" # ​:codex-file-citation[codex-file-citation]{line_range_start=31 line_range_end=67 path=models/hrm/hrm_act_v1.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/hrm/hrm_act_v1.py#L31-L67"}​

...
if self.config.pos_encodings == "rope":
    self.rotary_emb = RotaryEmbedding(
        dim=self.config.hidden_size // self.config.num_heads,
        max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
        base=self.config.rope_theta)
elif self.config.pos_encodings == "learned":
    self.embed_pos = CastedEmbedding(...)
elif self.config.pos_encodings == "monster":
    self.monster_emb = MonsterEmbedding(
        head_dim=self.config.hidden_size // self.config.num_heads,
        max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
        base=self.config.monster_theta,
        top_delta=self.config.monster_top_delta,
        skip_prefix=self.config.skip_prefix,
        prefix_len=self.puzzle_emb_len,
        use_xy=self.config.monster_use_xy,
        grid_w=self.config.monster_grid_w)                    # ​:codex-file-citation[codex-file-citation]{line_range_start=132 line_range_end=156 path=models/hrm/hrm_act_v1.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/hrm/hrm_act_v1.py#L132-L156"}​
...
pos_obj = None
if hasattr(self, "rotary_emb"):
    pos_obj = self.rotary_emb()
    if self.config.pos_encodings == "rope" and self.config.skip_prefix and self.puzzle_emb_len > 0:
        cos, sin = pos_obj; cos = cos.clone(); sin = sin.clone()
        k = int(self.puzzle_emb_len); cos[:k].fill_(1.0); sin[:k].zero_()
        pos_obj = (cos, sin)
elif hasattr(self, "monster_emb"):
    pos_obj = self.monster_emb()
seq_info = dict(cos_sin=pos_obj)                               # ​:codex-file-citation[codex-file-citation]{line_range_start=208 line_range_end=227 path=models/hrm/hrm_act_v1.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/hrm/hrm_act_v1.py#L208-L227"}​
Dataset & dataloading (Sudoku-focused)
dataset/build_sudoku_dataset.py

class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"
    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0                                           # ​:codex-file-citation[codex-file-citation]{line_range_start=18 line_range_end=24 path=dataset/build_sudoku_dataset.py git_url="https://github.com/commotum/MonSTERs/blob/main/dataset/build_sudoku_dataset.py#L18-L24"}​

def shuffle_sudoku(board, solution):
    digit_map = np.pad(np.random.permutation(np.arange(1,10)), (1,0))
    transpose_flag = np.random.rand() < 0.5
    bands  = np.random.permutation(3)
    row_perm = np.concatenate([b*3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s*3 + np.random.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i//9]*9 + col_perm[i%9] for i in range(81)])
    def apply_transformation(x):
        if transpose_flag: x = x.T
        new_board = x.flatten()[mapping].reshape(9,9).copy()
        return digit_map[new_board]
    return apply_transformation(board), apply_transformation(solution)  # ​:codex-file-citation[codex-file-citation]{line_range_start=27 line_range_end=57 path=dataset/build_sudoku_dataset.py git_url="https://github.com/commotum/MonSTERs/blob/main/dataset/build_sudoku_dataset.py#L27-L57"}​

def convert_subset(set_name, config):
    with open(hf_hub_download(config.source_repo,f"{set_name}.csv",repo_type="dataset"),newline="") as csvfile:
        ...
    num_augments = config.num_aug if set_name=="train" else 0
    results = {k: [] for k in ["inputs","labels","puzzle_identifiers","puzzle_indices","group_indices"]}
    ...
    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0: inp, out = orig_inp, orig_out
            else:            inp, out = shuffle_sudoku(orig_inp, orig_out)
            results["inputs"].append(inp); results["labels"].append(out)
            example_id += 1; puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }                                                          # ​:codex-file-citation[codex-file-citation]{line_range_start=60 line_range_end=127 path=dataset/build_sudoku_dataset.py git_url="https://github.com/commotum/MonSTERs/blob/main/dataset/build_sudoku_dataset.py#L60-L127"}​
dataset/common.py

DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    sets: List[str]

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    if tid == 0: return arr  # identity
    elif tid == 1: return np.rot90(arr, k=1)
    elif tid == 2: return np.rot90(arr, k=2)
    elif tid == 3: return np.rot90(arr, k=3)
    elif tid == 4: return np.fliplr(arr)
    elif tid == 5: return np.flipud(arr)
    elif tid == 6: return arr.T
    elif tid == 7: return np.fliplr(np.rot90(arr, k=1))
    else: return arr                                            # ​:codex-file-citation[codex-file-citation]{line_range_start=7 line_range_end=47 path=dataset/common.py git_url="https://github.com/commotum/MonSTERs/blob/main/dataset/common.py#L7-L47"}​
puzzle_dataset.py

def _sample_batch(rng, group_order, puzzle_indices, group_indices, start_index, global_batch_size):
    while (start_index < group_order.size) and (current_size < global_batch_size):
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        ...
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))
    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)  # ​:codex-file-citation[codex-file-citation]{line_range_start=14 line_range_end=38 path=puzzle_dataset.py git_url="https://github.com/commotum/MonSTERs/blob/main/puzzle_dataset.py#L14-L38"}​

class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int; dataset_path: str; global_batch_size: int; test_set_mode: bool
    epochs_per_iter: int; rank: int; num_replicas: int                           # ​:codex-file-citation[codex-file-citation]{line_range_start=41 line_range_end=50 path=puzzle_dataset.py git_url="https://github.com/commotum/MonSTERs/blob/main/puzzle_dataset.py#L41-L50"}​

def _lazy_load_dataset(self):
    field_mmap_modes = {"inputs": "r", "labels": "r", "puzzle_identifiers": None,
                        "puzzle_indices": None, "group_indices": None}
    self._data = {set_name: {field_name: np.load(os.path.join(self.config.dataset_path,self.split,
                        f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                        for field_name, mmap_mode in field_mmap_modes.items()}
                  for set_name in self.metadata.sets}                              # ​:codex-file-citation[codex-file-citation]{line_range_start=72 line_range_end=94 path=puzzle_dataset.py git_url="https://github.com/commotum/MonSTERs/blob/main/puzzle_dataset.py#L72-L94"}​

def _collate_batch(self, batch):
    batch = {k: v.astype(np.int32) for k, v in batch.items()}
    if self.metadata.ignore_label_id is not None:
        batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID
    if batch["puzzle_identifiers"].size < self.local_batch_size:
        pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
        pad_values = {"inputs": self.metadata.pad_id,
                      "labels": IGNORE_LABEL_ID,
                      "puzzle_identifiers": self.metadata.blank_identifier_id}
        batch = {k: np.pad(v, ((0, pad_size),)+((0,0),)*(v.ndim-1), constant_values=pad_values[k]) for k,v in batch.items()}
    return {k: torch.from_numpy(v) for k, v in batch.items()}                      # ​:codex-file-citation[codex-file-citation]{line_range_start=95 line_range_end=116 path=puzzle_dataset.py git_url="https://github.com/commotum/MonSTERs/blob/main/puzzle_dataset.py#L95-L116"}​

def __iter__(self):
    worker_info = get_worker_info()
    assert worker_info is None or worker_info.num_workers == 1
    self._lazy_load_dataset()
    if self.config.test_set_mode: yield from self._iter_test()
    else: yield from self._iter_train()                                            # ​:codex-file-citation[codex-file-citation]{line_range_start=189 line_range_end=199 path=puzzle_dataset.py git_url="https://github.com/commotum/MonSTERs/blob/main/puzzle_dataset.py#L189-L199"}​
Training/config glue
config/arch/hrm_v1.yaml

name: hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1
loss:
  name: losses@ACTLossHead
  loss_type: stablemax_cross_entropy
halt_exploration_prob: 0.1
halt_max_steps: 16
H_cycles: 2
L_cycles: 2
H_layers: 4
L_layers: 4
hidden_size: 768
num_heads: 8  # min(2, hidden_size // 64)
expansion: 4
puzzle_emb_ndim: ${.hidden_size}
# options: rope, learned, monster
pos_encodings: rope
skip_prefix: true                                            # ​:codex-file-citation[codex-file-citation]{line_range_start=1 line_range_end=23 path=config/arch/hrm_v1.yaml git_url="https://github.com/commotum/MonSTERs/blob/main/config/arch/hrm_v1.yaml#L1-L23"}​
pretrain.py

def create_dataloader(config, split, rank, world_size, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed, dataset_path=config.data_path,
        rank=rank, num_replicas=world_size, **kwargs), split=split)
    dataloader = DataLoader(
        dataset, batch_size=None,
        num_workers=1, prefetch_factor=8,
        pin_memory=True, persistent_workers=True)
    return dataloader, dataset.metadata                         # ​:codex-file-citation[codex-file-citation]{line_range_start=85 line_range_end=106 path=pretrain.py git_url="https://github.com/commotum/MonSTERs/blob/main/pretrain.py#L85-L106"}​

with torch.device("cuda"):
    model: nn.Module = model_cls(model_cfg)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
    if "DISABLE_COMPILE" not in os.environ:
        model = torch.compile(model, dynamic=False)             # ​:codex-file-citation[codex-file-citation]{line_range_start=125 line_range_end=129 path=pretrain.py git_url="https://github.com/commotum/MonSTERs/blob/main/pretrain.py#L125-L129"}​
3. Call Graphs & Data Shapes
Stage	Call/Operation	Shapes
Token load	PuzzleDataset.__iter__ → _sample_batch → _collate_batch	inputs, labels, puzzle_identifiers → [B,81] sequences
Embedding	_input_embeddings adds optional puzzle embeddings (prefix) then tokens	[B, prefix+T, H] where prefix = ceil(puzzle_emb_ndim/hidden_size)
Positional cache	RotaryEmbedding()/MonsterEmbedding()	cos/sin or ch/sh/cx/... cached with max_position_embeddings=prefix+seq_len
Attention	Attention.forward	hidden [B, T, H, D] → qkv [B, T, H, D]; head_dim D split: main (D//12)*12 → triads [... ,F,12], tail passthrough
Position apply	apply_rotary_pos_emb or apply_monster_pos_emb	broadcast over [T, head_dim] or [T, F] tables; no per-token loops
FlashAttention	flash_attn_func	consumes [B,H,T,D] Q,K,V; returns [B,H,T,D]
Flatten ↔ grid mapping: MonsterEmbedding derives x = idx % grid_w, y = idx // grid_w, sub-grid z for 3×3 boxes via grid_w ** 0.5. shuffle_sudoku maps (i,j) to flattened index via row_perm[i//9]*9 + col_perm[i%9].

4. Efficiency Notes
Precomputed caches: RotaryEmbedding and MonsterEmbedding store cos/sin or cosh/sinh tables in buffers for reuse; skip_prefix zeros out first prefix_len to avoid recompute.

Vectorized operations: apply_rotary_pos_emb/apply_monster_pos_emb use broadcasting and reshape rather than loops; triad split avoids per-frequency iteration.

Contiguous layout: Attention keeps [bs, seq_len, heads, head_dim] contiguous and only reshapes once for FlashAttention, avoiding transposes.

Limited casting: Inputs cast once to match table dtype and restored after embedding to preserve precision.

FlashAttention + compile: flash_attn_func gives fused attention; torch.compile accelerates model graph.

DataLoader knobs: pin_memory, persistent_workers, prefetch_factor=8 keep IO throughput high; .npy loaded via memmap to avoid copies.

Assumptions: head_dim % 12 → triads with Minkowski mask; fixed [B,T,…] layout; Sudoku seq_len fixed at 81 for flattening.

5. Fragility & Sequence Coupling
Position tables indexed purely by sequence ID (pos), assuming flat [T]; MonsterEmbedding reconstructs (x,y,z) using grid_w and constant square root, tied to 9×9 Sudoku grids.

apply_monster_pos_emb slices head_dim into blocks of 12; non-multiples are silently passed through but still assume triads order.

PuzzleDataset and training code rely on .npy files (inputs.npy, labels.npy, puzzle_indices.npy, group_indices.npy) and shape constants (seq_len=81) baked into metadata.

RoPE skip-prefix logic explicitly edits first prefix_len rows of [T,F] cache, tightly coupling to sequential ordering.

6. Hook Points for ND Refactor
Dataset → model: Extend _collate_batch to emit a pos_dict alongside tokens; modify PuzzleDataset return type.

Positional registry: Add factory mapping ("learned"|"rope"|"monster") to embedding classes; HierarchicalReasoningModel_ACTV1_Inner.__init__ already branches—replace with registry lookup.

Unified interface: Define PositionalEncoding.forward(q, k, pos_obj, head_dim); current apply_* functions provide reference implementations.

Cache management: Keep buffers (cos/sin or ch/sh/...) on device; derive pos_obj from pos_dict instead of implicit indices.

Sequence-agnostic mask: Replace Minkowski pattern12 with runtime mask derived from triad layout; allow tail dims per axis count.

Data pipeline: Update .npy writer to save pos_dict arrays (or memmaps) per axis; collate maintains dtypes/devices.

7. Minimal Migration Plan
Dataset builder outputs (tokens, pos_dict, meta); .npy or memmap updated to include per-axis arrays.

Collate function threads pos_dict through DataLoader; padding aligns each axis.

Positional encoding registry selects learned|rope|monster and constructs per-device caches.

Attention modules call a unified apply_pos_emb(q,k,pos_obj,head_dim) interface.

Preserve FlashAttention path and DataLoader settings; ensure caches are built once per device.

Sequence assumptions replaced by {t,x,y,z,...} indexing; MonsterEmbedding generalized to read positions rather than compute from flat idx.


---


v4

Task 1: Code Extraction
A. Positional Encoding Methods
RoPE

# models/layers.py L30-L40, L260-L275
def apply_rotary_pos_emb(q, k, cos, sin):
    q = q.to(cos.dtype);  k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)
    def forward(self):
        return self.cos_cached, self.sin_cached
MonSTER

# models/layers.py L48-L151, L154-L220
class MonsterEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings, base=10000.0,
                 top_delta=9, skip_prefix=True, prefix_len=0,
                 use_xy=True, grid_w=9, device=None):
        ...
        pos = torch.arange(self.max_pos, dtype=torch.float32, device=device)
        if self.use_xy:
            idx = torch.clamp(pos - self.prefix_len, min=0)
            y = (idx // self.grid_w).to(torch.float32)
            x = (idx % self.grid_w).to(torch.float32)
            sub_grid = int(self.grid_w ** 0.5)
            z = torch.floor(x / sub_grid) + torch.floor(y / sub_grid) * sub_grid
        ...
        self.ch = nn.Buffer(ch, persistent=False);  self.sh = nn.Buffer(sh, persistent=False)
        self.cx = nn.Buffer(cx, persistent=False);  self.sx = nn.Buffer(sx, persistent=False)
        self.cy = nn.Buffer(cy, persistent=False);  self.sy = nn.Buffer(sy, persistent=False)
        self.cz = nn.Buffer(cz, persistent=False);  self.sz = nn.Buffer(sz, persistent=False)

def apply_monster_pos_emb(q, k, tables, head_dim):
    if tables.get("num_freq", None) == 0:  return q, k
    ch, sh = tables["ch"], tables["sh"];  cx, sx = tables["cx"], tables["sx"]
    cy, sy = tables["cy"], tables["sy"];  cz, sz = tables["cz"], tables["sz"]
    ...
    main = x[..., :main_dim].view(bs, T, H, main_dim // 12, 12)
    ...
    return _apply(q), _apply(k)
Learned Positional Encoding

# models/layers.py L242-L257
class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, init_std, cast_to):
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
    def forward(self, input):
        return F.embedding(input, self.embedding_weight.to(self.cast_to))
Integration & Selection Logic

# models/hrm/hrm_act_v1.py L132-L157, L173-L194, L207-L226
# __init__: choose encoding
if self.config.pos_encodings == "rope":
    self.rotary_emb = RotaryEmbedding(...)
elif self.config.pos_encodings == "learned":
    self.embed_pos = CastedEmbedding(...)
elif self.config.pos_encodings == "monster":
    self.monster_emb = MonsterEmbedding(...)
else:
    raise NotImplementedError()

# _input_embeddings: apply learned positional embeddings
if self.config.pos_encodings == "learned":
    embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

# forward: build positional object; skip puzzle prefix if configured
pos_obj = None
if hasattr(self, "rotary_emb"):
    pos_obj = self.rotary_emb()
    if self.config.pos_encodings == "rope" and self.config.skip_prefix and self.puzzle_emb_len > 0:
        cos, sin = pos_obj; k = int(self.puzzle_emb_len)
        cos[:k].fill_(1.0); sin[:k].zero_(); pos_obj = (cos, sin)
elif hasattr(self, "monster_emb"):
    pos_obj = self.monster_emb()
seq_info = dict(cos_sin=pos_obj)
# models/layers.py L301-L329
class Attention(nn.Module):
    ...
    def forward(self, cos_sin, hidden_states):
        ...
        # Positional encoding: RoPE or MonSTER
        if cos_sin is not None:
            if isinstance(cos_sin, tuple):
                cos, sin = cos_sin
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
            elif isinstance(cos_sin, dict) and cos_sin.get("kind") == "monster":
                query, key = apply_monster_pos_emb(query, key, cos_sin, self.head_dim)
                key = key * self.minkowski_mask.to(key.dtype)
        ...
B. Dataloading
# pretrain.py L85-L106
def create_dataloader(config, split, rank, world_size, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata
# puzzle_dataset.py key sections
class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int
    rank: int
    num_replicas: int

class PuzzleDataset(IterableDataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()
        assert self.config.global_batch_size % self.config.num_replicas == 0
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas
        self._data = None
        self._iters = 0

    def _lazy_load_dataset(self):
        if self._data is not None: return
        field_mmap_modes = {"inputs": "r", "labels": "r",
                            "puzzle_identifiers": None, "puzzle_indices": None,
                            "group_indices": None}
        self._data = {set_name: {field_name: np.load(...)}
                      for set_name in self.metadata.sets}

    def _iter_train(self):
        ...
        group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1)
                                      for _ in range(self.config.epochs_per_iter)])
        while start_index < group_order.size:
            start_index, batch_indices, batch_puzzle_indices = _sample_batch(...)
            ...
            yield set_name, batch, global_effective_batch_size

    def _iter_test(self):
        ...
        while start_index < total_examples:
            ...
            yield set_name, batch, end_index - start_index

    def __iter__(self):
        self._lazy_load_dataset()
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
C. Sudoku Dataset Creation and Augmentation
# dataset/build_sudoku_dataset.py L18-L24, L27-L57, L60-L160
class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"
    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0

def shuffle_sudoku(board, solution):
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))    # digit shuffling
    transpose_flag = np.random.rand() < 0.5                                # optional transpose
    bands = np.random.permutation(3)                                       # permute row bands
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)                                      # permute column stacks
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])
    def apply_transformation(x):
        if transpose_flag: x = x.T
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        return digit_map[new_board]
    return apply_transformation(board), apply_transformation(solution)

def convert_subset(set_name, config):
    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:
        ...
        inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
        labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    if set_name == "train" and config.subsample_size is not None:
        ...
    num_augments = config.num_aug if set_name == "train" else 0

    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)  # augmentation

            results["inputs"].append(inp)
            results["labels"].append(out)
            ...

    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)  # flatten 9x9 -> 81
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        ...
    }

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)
Command-line flags --subsample-size, --min_difficulty, and --num-aug come directly from DataProcessConfig.

Task 2: Analysis
Flatten‑then‑Apply Strategy
The dataset builder reads each Sudoku grid, converts characters to integers, and flattens the 9×9 board into an 81-element sequence (arr = np.concatenate(seq).reshape(len(seq), -1)), adding 1 so that 0 can serve as the padding token. During training, create_dataloader loads batches of these sequences. In HierarchicalReasoningModel_ACTV1_Inner, positional encoding tables are generated (self.rotary_emb() or self.monster_emb()), possibly skipping the puzzle‑identifier prefix, and Attention.forward applies the chosen scheme (apply_rotary_pos_emb or apply_monster_pos_emb) to the query/key tensors. Thus, an entire Sudoku puzzle is treated as a single 81‑token 1D sequence before RoPE/MonSTER is applied.

Efficiency for Sequences

The dataset is pre-flattened and stored as contiguous arrays, allowing vectorized loading and collation in PuzzleDataset.

RoPE and MonSTER precompute cosine/sine or hyperbolic tables (cos_cached, ch, cx, etc.) and cache them as buffers, so positional information is a lookup rather than computed per step.

Attention operations remain standard FlashAttention calls over [batch, seq_len, num_heads, head_dim] tensors, enabling GPU-optimized kernels and compatibility with existing sequence models.

How Attention Learns 2D Structure (Hypothesis)
Although the positional encoding is purely 1D, the network is trained on thousands of augmented puzzles where structural relationships (rows, columns, 3×3 boxes) correspond to constant offsets in the flattened sequence. For example, cells in the same column are always offset by multiples of 9; box neighbors have characteristic offset patterns (e.g., ±1, ±9, ±10). Self-attention heads can specialize to attend to tokens at these fixed strides, effectively reconstructing the grid topology. Over many samples with shuffled digits, row/column permutations, and optional transpose, the model observes that certain relative positions consistently share strong logical constraints. This encourages individual heads to detect and exploit these offsets, enabling the transformer to re-discover Sudoku’s 2D structure even though the data is presented as a 1D sequence.

---

v5

Positional Encodings
apply_rotary_pos_emb rotates query/key halves and multiplies by cached cos/sin tables, returning mixed-dtype tensors for efficiency

MonsterEmbedding precomputes MonSTER scalar tables (cosh/sinh/xyz cos-sin) with options for skipping prefixes and mapping 1D positions back to Sudoku grid coordinates

apply_monster_pos_emb slices heads into 12‑dim blocks and applies the precomputed tables in a fully vectorized manner before concatenation

RotaryEmbedding caches RoPE cos/sin tensors for all positions

Attention selects between RoPE and MonSTER via apply_* helpers, then calls FlashAttention for batched matrix products

HRM configuration enables "learned", "rope", or "monster" positional encodings; learned embeddings come from embed_pos while RoPE/MonSTER are instantiated at construction

_input_embeddings combines token, optional puzzle, and positional embeddings (scaled by √hidden_size)

In forward, position tables are fetched once per batch; prefix tokens can be zeroed for RoPE, and MonSTER tables are passed to attention blocks through seq_info

Data Loading
PuzzleDatasetConfig and PuzzleDataset implement distributed iterable loading with global/local batch sizes and rank partitioning

Datasets are lazily memory‑mapped (np.load(..., mmap_mode='r')) to avoid loading everything into RAM

Batches are collated by converting dtype, remapping ignore IDs, padding to local batch size, and returning tensors

Training iteration shuffles group IDs, draws puzzle examples without replacement, and yields per-rank slices—no Python loops over individual samples after sampling indices

create_dataloader wraps PuzzleDataset with a non‑batching DataLoader that uses one worker, high prefetch_factor, and pinned memory for GPU throughput

Dataset Creation & Augmentation
PuzzleDatasetMetadata records token IDs, sequence length, vocabulary, and dataset grouping stats

dihedral_transform enumerates eight grid symmetries for tasks needing dihedral augmentation; inverse mapping is provided

Sudoku preprocessing downloads CSVs, filters by difficulty, optionally subsamples, and applies shuffle_sudoku augmentations (digit remapping, band/stack row‑column permutations, optional transpose)

Each puzzle/solution pair is flattened and concatenated into arrays; indices mark puzzle/group boundaries and identifiers. Arrays are stored in .npy files with accompanying JSON metadata

Efficiency Analysis
Vectorization & Broadcasting: Positional encoding helpers operate on entire tensors, relying on broadcasting and reshaping instead of explicit loops (e.g., MonSTER’s _apply slices 12‑dim triads at once).

Caching: Rotary and MonSTER embeddings precompute and cache cos/sin tables, eliminating per-step trigonometric overhead and enabling fast reuse across batches.

Memory Mapping & Prefetching: Puzzle datasets are memory‑mapped, avoiding heavy RAM usage, while the DataLoader uses pinned memory and prefetching to overlap I/O with GPU compute.

Batch Sampling Without Loops: _sample_batch builds batches by manipulating index arrays; _iter_train shuffles group order and uses vectorized slicing, minimizing Python overhead even with varying puzzle lengths.

Flattening Spatial Grids: Sudoku boards (9×9) are flattened to 1D sequences before saving, enabling uniform tokenized arrays for batched processing, but requiring positional encodings (RoPE/MonSTER) to reconstruct or infer original 2D relations during attention.

The overall design—vectorized positional embeddings, memory-mapped datasets, and sequence‑level batching—makes the system efficient for large‐scale training on flattened grid data while allowing attention mechanisms to learn underlying spatial structure from positional cues.

---

v6

