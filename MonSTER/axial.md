## Executive summary: what’s wrong with axial RoPE

Axial RoPE is fundamentally limited because it factorizes multi-dimensional position into independent one-dimensional rotations. As a result, it cannot represent mixed spatial directions (e.g. diagonals) intrinsically, and all positional interactions are axis-separable.

---

## 1. Definition of axial RoPE

For a 2D position
[
p = (x, y),
]
axial RoPE applies independent 1D RoPE rotations along each axis:

[
R(x,y)
======

\begin{pmatrix}
R_x(x) & 0 \
0 & R_y(y)
\end{pmatrix},
]

where each
[
R_x, R_y \in SO(2)^{d/4}
]
is a block-diagonal rotation matrix acting on disjoint channel groups.

Queries and keys are transformed as:
[
q' = R(x_q, y_q), q, \quad
k' = R(x_k, y_k), k.
]

The resulting attention inner product is:
[
\langle q', k' \rangle
======================

\langle q_x, R_x(\Delta x), k_x \rangle
+
\langle q_y, R_y(\Delta y), k_y \rangle,
]
where
[
\Delta x = x_q - x_k, \quad
\Delta y = y_q - y_k.
]

---

## 2. Root problem: block-diagonal separability

The rotation matrix is block-diagonal across axes. This has two immediate consequences:

* Each complex rotation pair encodes **only one axis**.
* No positional feature ever depends jointly on both (x) and (y).

Formally, the phase of each RoPE channel satisfies:
[
\phi_i(x,y) \in {\theta_i^x x,; \theta_i^y y},
]
but **never**
[
\theta_i^x x + \theta_i^y y.
]

Thus, positional phase gradients lie in the restricted set:
[
{(c,0),; (0,c)} \subset \mathbb{R}^2,
]
rather than arbitrary directions ((c_1, c_2)).

---

## 3. Why diagonal directions are fundamentally weak

Consider two relative displacements:

* Axis-aligned: ((\Delta x, 0))
* Diagonal: ((\Delta x, \Delta x))

In axial RoPE:

* Axis-aligned displacement affects only x-channels.
* Diagonal displacement affects x- and y-channels **independently**.

Crucially, there is **no channel** whose phase is proportional to:
[
\Delta x + \Delta y.
]

Diagonal structure is therefore not a primitive direction in the representation. It can only be reconstructed later by linear mixing of unrelated axis-specific channels.

By contrast, mixed RoPE variants (e.g. MonSTER-style encodings) include channels where:
[
\phi(\Delta x, \Delta y) = \theta_x \Delta x + \theta_y \Delta y,
]
making diagonal sensitivity **intrinsic**, not emergent.

---

## 4. Why attention cannot “fix this later”

A subtle but critical point:

RoPE already uses the query–key dot product to convert absolute position into **relative position**, via:
[
R(p_q)^\top R(p_k) = R(p_q - p_k).
]

This consumes the only multiplicative interaction between positional phases.

After this step:

* There is no remaining mechanism for x- and y-phases to interact multiplicatively.
* Any axis mixing must occur **after attention**, through learned linear projections.

Such post-hoc mixing is:

* weaker,
* more data-hungry,
* and not position-equivariant.

---

## 5. Lack of spatial hierarchy

Axial RoPE encodes only pairwise offsets. It provides no mechanism for:

* grouping patches into regions,
* distinguishing “same object” vs. “different object,”
* representing coarse-to-fine spatial structure.

Mathematically, all spatial information is collapsed into:
[
R(\Delta x, \Delta y),
]
with no notion of composition, nesting, or multi-scale organization.