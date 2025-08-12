# V1

---

HierarchicalReasoningModel_ACTV1Config hard‑codes a string enum for the three positional encoding modes—learned, rope, and monster

Initialization chooses among RotaryEmbedding, a learned embedding table, or MonsterEmbedding via a series of if/elif branches, embedding each case directly inside the model constructor

Learned positional embeddings are added ad‑hoc in _input_embeddings, mixing token, puzzle, and positional embeddings with manual scaling

The forward pass inspects attributes (hasattr(self, "rotary_emb")) and mutates RoPE tables to skip a prefix, whereas MonSTER has its own call path—an inconsistent and somewhat “hacked” approach

The attention layer accepts a cos_sin argument that may be a tuple or a dict, requiring type checks to decide whether to apply RoPE or MonSTER rotations

RotaryEmbedding is compact and self‑contained

MonsterEmbedding precomputes large tables with additional prefix handling, embedding all logic in a single class

apply_monster_pos_emb is a lengthy standalone function that manipulates tables directly rather than through an object interface

Recommendations

Create a dedicated positional‑encoding module (e.g., models/positional.py) and move RotaryEmbedding, MonsterEmbedding, and learned embedding helpers out of the monolithic layers.py.

Unify the API: each encoding class should expose a standard interface like PositionalEncoding.apply(q, k) and PositionalEncoding.prepare(seq_len, device) so the attention layer never inspects types or attributes.

Handle prefix skipping inside the encoding classes, removing RoPE’s manual cloning and zeroing in the model’s forward path.

Separate embedding creation from application: _input_embeddings should delegate positional logic to the encoding object instead of mixing token/puzzle/pos embeddings manually.

Factor out shared utilities (e.g., rotation helpers, caching) to avoid duplication and make it easier to add new encodings.

Best Path Forward (expert view)

Design a base PositionalEncoding class with hooks for precomputation, prefix handling, and incremental generation; subclass it for RoPE, learned, and MonSTER.

Refactor Attention to accept a single encoding object and call encoding.apply(q, k)—this removes if isinstance checks and encourages plug‑and‑play experimentation with new schemes.

Provide configuration builders or factory functions so models can request an encoding without embedding all choices in their constructors.

Consider unit tests and benchmarks for each encoding type to ensure correctness and performance parity after refactoring.

Once unified, explore additional encodings or hybrid strategies, confident that the abstraction won’t require touching model internals.

---

# V2

---

## 1. Repo tree (code‑only)

├── .gitignore
├── .gitmodules
├── .python-version
├── .vscode
│ ├── launch.json
│ └── settings.json
├── LICENSE
├── MonSTER
│ ├── 4 Dimension Attention.md
│ ├── 4D.md
│ ├── MonSTER.py
│ ├── create_arc_mini.sh
│ ├── hrm-github.txt
│ ├── monsters-github.txt
│ ├── note_free.py
│ └── token_analysis.py
├── README.md
├── assets
│ └── npyjs.js
├── config
│ ├── arch
│ │ └── hrm_v1.yaml
│ └── cfg_pretrain.yaml
├── dataset
│ ├── build_arc_dataset.py
│ ├── build_maze_dataset.py
│ ├── build_sudoku_dataset.py
│ └── common.py
├── evaluate.py
├── flash-attention
├── models
│ ├── common.py
│ ├── hrm
│ │ └── hrm_act_v1.py
│ ├── layers.py
│ ├── losses.py
│ └── sparse_embedding.py
├── pretrain.py
├── puzzle_dataset.py
├── puzzle_visualizer.html
├── requirements.txt
└── utils
└── functions.py

python
Copy

---

## 2. Positional encodings — code & file locations

### RoPE

**`models/layers.py`**​:codex-file-citation[codex-file-citation]{line_range_start=23 line_range_end=40 path=models/layers.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/layers.py#L23-L40"}​​:codex-file-citation[codex-file-citation]{line_range_start=260 line_range_end=275 path=models/layers.py git_url="https://github.com/commotum/MonSTERs/blob/main/models/layers.py#L260-L275"}​
```python
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
Integration — models/hrm/hrm_act_v1.py

python
Copy
if self.config.pos_encodings == "rope":
    self.rotary_emb = RotaryEmbedding(
        dim=self.config.hidden_size // self.config.num_heads,
        max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
        base=self.config.rope_theta,
    )
Application — models/layers.py (Attention)

python
Copy
if isinstance(cos_sin, tuple):
    cos, sin = cos_sin
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
Learned positional embedding
**models/layers.py**

python
Copy
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
Selection — models/hrm/hrm_act_v1.py

python
Copy
elif self.config.pos_encodings == "learned":
    self.embed_pos = CastedEmbedding(
        self.config.seq_len + self.puzzle_emb_len,
        self.config.hidden_size,
        init_std=embed_init_std,
        cast_to=self.forward_dtype,
    )
Application — _input_embeddings in models/hrm/hrm_act_v1.py

python
Copy
if self.config.pos_encodings == "learned":
    # scale by 1/sqrt(2) to maintain forward variance
    embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
MonSTER
models/layers.py — MonsterEmbedding & helper

python
Copy
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
models/layers.py — apply_monster_pos_emb

python
Copy
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
Helper constant — Minkowski mask in Attention.__init__

python
Copy
pattern12 = torch.tensor([1, -1, -1, -1] * 3, dtype=torch.float32)
main_dim = (self.head_dim // 12) * 12
mask = torch.ones(self.head_dim, dtype=torch.float32)
if main_dim > 0:
    F = main_dim // 12
    mask[:main_dim] = pattern12.repeat(F)
mask = mask.view(1, 1, 1, self.head_dim)
self.register_buffer("minkowski_mask", mask, persistent=False)
Selection — models/hrm/hrm_act_v1.py

python
Copy
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
Application — models/layers.py (Attention)

python
Copy
elif isinstance(cos_sin, dict) and cos_sin.get("kind") == "monster":
    query, key = apply_monster_pos_emb(query, key, cos_sin, self.head_dim)
    key = key * self.minkowski_mask.to(key.dtype)
3. Refactor recommendation
Target structure

bash
Copy
models/positional/
  base.py        # PositionalEncoding interface
  rope.py        # RotaryEmbedding + apply_rotary_pos_emb
  learned.py     # CastedEmbedding + add-to-input helper
  monster.py     # MonsterEmbedding + apply_monster_pos_emb + Minkowski mask
  __init__.py    # registry/factory: build_positional(name, **cfg)
Minimal interface

python
Copy
class PositionalEncoding(nn.Module):
    def precompute(self, seq_len: int, device, dtype): ...
    def apply_inputs(self, emb, tables): return emb  # override in learned
    def apply_qk(self, q, k, tables): return q, k     # override in rope/monster
Factory/registry

python
Copy
def build_positional(name: str, **cfg) -> PositionalEncoding:
    return {"learned": LearnedPE, "rope": RopePE, "monster": MonsterPE}[name](**cfg)
Call sites to update

models/hrm/hrm_act_v1.py: replace conditional block (lines 132‑155) with self.pos_emb = build_positional(config.pos_encodings, ...).

models/hrm/hrm_act_v1.py::_input_embeddings: call self.pos_emb.apply_inputs(embedding, tables) instead of manual addition (lines 187‑190).

models/layers.py::Attention.forward: use query, key = self.pos_emb.apply_qk(query, key, tables) and move Minkowski mask handling into MonsterPE.

models/hrm/hrm_act_v1.py::forward: obtain tables via tables = self.pos_emb.precompute(...) instead of ad‑hoc rotary_emb() / monster_emb().

Invariants / assumptions

MonSTER requires head_dim % 12 == 0 to form triads; validate in MonsterPE.__init__.

Minkowski mask multiplies keys after MonSTER rotation; placing it inside MonsterPE.apply_qk keeps attention code generic.

Learned embedding scales inputs by 1/√2; retain this scaling in LearnedPE.apply_inputs.

---

V3

---

1. Repo tree (code-only)
.
  LICENSE
  .gitmodules
  README.md
  requirements.txt
  puzzle_dataset.py
  evaluate.py
  pretrain.py
  .gitignore
  .python-version
  puzzle_visualizer.html
  dataset/
    build_sudoku_dataset.py
    build_maze_dataset.py
    build_arc_dataset.py
    common.py
  config/
    cfg_pretrain.yaml
    arch/
      hrm_v1.yaml
  .vscode/
    launch.json
    settings.json
  utils/
    functions.py
  MonSTER/
    create_arc_mini.sh
    4 Dimension Attention.md
    note_free.py
    token_analysis.py
    monsters-github.txt
    hrm-github.txt
    MonSTER.py
    4D.md
  assets/
    npyjs.js
  models/
    layers.py
    losses.py
    sparse_embedding.py
    common.py
    hrm/
      hrm_act_v1.py
  flash-attention/

2. Positional encodings — code + file locations (RoPE, Learned, MonSTER)
RoPE

**models/layers.py**

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
**models/layers.py**

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
Learned

**models/layers.py**

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
**models/hrm/hrm_act_v1.py**

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
MonSTER

**models/layers.py**

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
**models/layers.py**

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
**models/layers.py**

        pattern12 = torch.tensor([1, -1, -1, -1] * 3, dtype=torch.float32)
        main_dim = (self.head_dim // 12) * 12
        mask = torch.ones(self.head_dim, dtype=torch.float32)
        if main_dim > 0:
            F = main_dim // 12
            mask[:main_dim] = pattern12.repeat(F)
        mask = mask.view(1, 1, 1, self.head_dim)
        self.register_buffer("minkowski_mask", mask, persistent=False)
Integration point

**models/layers.py**

# Positional encoding: RoPE or MonSTER
if cos_sin is not None:
    if isinstance(cos_sin, tuple):
        cos, sin = cos_sin
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
    elif isinstance(cos_sin, dict) and cos_sin.get("kind") == "monster":
        query, key = apply_monster_pos_emb(query, key, cos_sin, self.head_dim)
        key = key * self.minkowski_mask.to(key.dtype)
3. Refactor recommendation (align with plan)
Package layout

Create models/positional/ with:

base.py – PositionalEncodingBase with build_tables and apply.

rope.py, learned.py, monster.py implementing the interface.

__init__.py exposing get_positional(name, **cfg).

Unified interface

build_tables(pos_dict, *, max_len, head_dim, device) -> pos_obj

apply(q, k, pos_obj, *, head_dim) -> (q, k)

Learned backend returns embedding table from build_tables and no-op apply.

Hook points

puzzle_dataset.py – thread pos_dict through _lazy_load_dataset and _collate_batch so batches yield {'inputs', 'labels', 'puzzle_identifiers', 'pos_dict'}

models/hrm/hrm_act_v1.py

Replace constructor branching with self.pos_enc = get_positional(config.pos_encodings, **cfg)

In _input_embeddings, delegate to self.pos_enc for learned tables; keep vectorized path

In forward, call pos_obj = self.pos_enc.build_tables(batch['pos_dict'], max_len=seq_len, head_dim=head_dim, device=z_H.device) and pass downstream.

models/layers.py::Attention.forward – replace branching with query, key = self.pos_enc.apply(query, key, pos_obj, head_dim=self.head_dim).

I/O & caching

Store pos_dict arrays as .npy/memmap alongside existing fields; enable DataLoader options (pinned memory, prefetch, persistent workers) for fast transfer.

Cache trig/hyperbolic tables per device/sequence length inside each positional module.

Preserve FlashAttention interface (flash_attn_func) and Minkowski mask multiplication for MonSTER.

Invariants & notes

Assume head_dim multiples of 12 for MonSTER triads; mask lives in models/layers.py::Attention.__init__.

pos_dict expected keys: {'t', 'x', 'y', 'z'}; modules must remain fully vectorized and respect existing skip-prefix behavior.