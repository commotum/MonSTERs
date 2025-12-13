# MonSTERs for Four-Dimension Attention
## Minkowski Space-Time Embedding Rotors for 4D Generalization and World Modeling

**Jacob Peterson**
*[peterj29@oregonstate.edu](mailto:peterj29@oregonstate.edu)*

---

## Abstract

Rotary Position Embedding (RoPE) was a practical breakthrough because it achieved **relative positional behavior** using only **absolute, modification-light transforms** applied to queries and keys. This “absolute→relative fusion” property makes RoPE drop-in compatible with architectures—such as linear attention variants—that cannot incorporate learned relative biases by directly modifying the attention matrix.

This document presents **MonSTERs** as a principled extension of RoPE from 1D Euclidean rotations to **4D Minkowski spacetime**. MonSTERs replaces RoPE’s SO(2) plane rotations with a structured action of the Lorentz group SO(1,3) over 4D blocks, preserving a Minkowski metric and retaining RoPE’s defining fusion identity under a Minkowski inner product:
[
\langle L(s_q)q,; L(s_k)k\rangle_\eta
=====================================

\langle q,; L(s_k-s_q)k\rangle_\eta.
]
Unlike axial RoPE variants that factorize multi-dimensional position into independent 1D rotations, MonSTERs encodes coupled spacetime structure intrinsically, enabling mixed-direction sensitivity (e.g., diagonals) within the core positional mechanism rather than requiring post-hoc learned mixing.

---

## 1. Introduction

Transformers are permutation-invariant in the absence of positional information: self-attention alone does not distinguish token order or geometry. Positional encoding is therefore not an optional refinement but a structural requirement.

RoPE addressed this requirement in a particularly robust way. Instead of introducing relative-position parameters into the attention matrix (which complicates integration and breaks compatibility with attention variants that do not explicitly form the attention matrix), RoPE applies an **absolute positional transform** to queries and keys such that the **inner product depends only on relative displacement**. This is the essential “absolute→relative fusion” insight.

MonSTERs follows the same trajectory: it preserves RoPE’s defining identity while generalizing the geometry from Euclidean rotation to Minkowski spacetime. The aim is not to “add physics,” but to extend the underlying group-theoretic mechanism that makes RoPE work to a richer setting where multi-axis interactions are not forced to be separable.

---

## 2. RoPE as absolute→relative fusion

### 2.1 The fusion objective

RoPE can be framed as selecting a family of absolute transforms (R(m)) indexed by position (m) such that, for transformed queries and keys
[
q' = R(m),q,\qquad k' = R(n),k,
]
the similarity depends only on the relative offset (n-m). In the standard Euclidean dot-product setting, the canonical identity is:
[
\langle R(m)q,; R(n)k\rangle
============================

\langle q,; R(n-m)k\rangle.
]

This property is not a convenient trick; it is the core design requirement. The reason it holds is structural:

1. **Group structure**: ({R(m)}) forms a group action with (R(m)^{-1}R(n)=R(n-m)).
2. **Metric preservation**: (R(m)) preserves the inner product (orthogonality).
3. **Additive parameterization**: the transform parameters are linear in position, so compositions reflect position differences.

MonSTERs will preserve these three ingredients, replacing Euclidean orthogonality with Lorentz (Minkowski) isometry.

---

## 3. Axial RoPE and the diagonal limitation

### 3.1 Definition of axial RoPE (2D)

For a 2D position (p=(x,y)), axial RoPE applies independent 1D rotary transforms along each axis:
[
R(x,y)=
\begin{pmatrix}
R_x(x) & 0\
0 & R_y(y)
\end{pmatrix},
]
where (R_x(x)) and (R_y(y)) are block-diagonal SO(2) rotations acting on disjoint channel groups.

With
[
q' = R(x_q,y_q),q,\quad
k' = R(x_k,y_k),k,
]
the attention inner product decomposes:
[
\langle q',k'\rangle
====================

\langle q_x,;R_x(\Delta x)k_x\rangle
+
\langle q_y,;R_y(\Delta y)k_y\rangle,
\qquad
\Delta x=x_q-x_k,;\Delta y=y_q-y_k.
]

### 3.2 Root issue: block-diagonal separability

Because (R(x,y)) is block-diagonal across axes, axial RoPE is **axis-separable**:

* each channel depends on **one** coordinate only,
* no channel phase depends jointly on ((x,y)).

Equivalently, the phase function for any rotary channel satisfies:
[
\phi_i(x,y)\in {\theta_i^x x,;\theta_i^y y},
\quad\text{but not}\quad
\theta_i^x x + \theta_i^y y.
]

This constrains representable phase gradients to axis-aligned directions. Diagonal structure ((\Delta x,\Delta x)) has no primitive representation channel whose phase directly tracks (\Delta x+\Delta y); it must be reconstructed later by linear mixing.

### 3.3 Why attention cannot “fix it later”

RoPE-style fusion consumes the key multiplicative interaction when forming:
[
R(p_q)^\top R(p_k)=R(p_k-p_q).
]
After this contraction, there is no remaining mechanism for multiplicative coupling between axis phases. Any diagonal recovery must occur after attention through learned projections, which is weaker, more data-hungry, and not intrinsically position-equivariant.

This motivates a coupled multi-dimensional group action in the positional mechanism itself.

---

## 4. MonSTERs: Lorentz-invariant RoPE in 4D Minkowski spacetime

### 4.1 Spacetime positions and Minkowski inner product

MonSTERs assigns each token an absolute spacetime position
[
s=(t,x,y,z)\in \mathbb{R}^4.
]
We use the Minkowski metric with signature ((+,-,-,-)):
[
\eta=\mathrm{diag}(1,-1,-1,-1),
]
and define the Minkowski inner product:
[
\langle u,v\rangle_\eta := u^\top \eta v.
]

### 4.2 The key idea: replace SO(2) with SO(1,3) while preserving fusion

RoPE applies orthogonal rotations (R(m)\in SO(d)) (blockwise) that preserve the Euclidean dot product. MonSTERs instead applies Lorentz transformations (L(s)\in SO(1,3)) (blockwise) that preserve the Minkowski inner product:
[
L(s)^\top \eta L(s)=\eta.
]
This is the precise Minkowski analogue of orthogonality in RoPE.

### 4.3 Embedding layout and frequency buckets (v12)

As in RoPE, MonSTERs uses multiple frequencies. The embedding channels are partitioned into **frequency buckets**, and each bucket is further subdivided into **three 4D blocks** (a triad). In the v12 structure, each frequency bucket therefore consumes 12 channels:
[
\text{slice} = 12 = [4 ;|; 4 ;|; 4].
]
This triad layout supports multiple coupled spacetime transforms per frequency while maintaining the “many small blocks” computational style of RoPE.

For frequency index (j), let (\lambda_j) denote the inverse-frequency scalar, typically following a geometric schedule (RoPE-style).

### 4.4 Absolute parameterization (v12): linear in position

For each frequency (\lambda_j), MonSTERs defines a set of scalar transform parameters as **linear functions of the absolute spacetime coordinates**:
[
\phi_j(s)=\lambda_j,u,t,
\qquad
\theta^x_j(s)=\lambda_j,u,x,
\qquad
\theta^y_j(s)=\lambda_j,u,y,
\qquad
\theta^z_j(s)=\lambda_j,u,z,
]
where (u) is a global scaling factor chosen so that angles/rapidities stay in a numerically stable range.

This mirrors RoPE’s defining feature: rotation phases are linear in position. Here, rapidity and spatial angles are linear in spacetime coordinates.

### 4.5 The per-block Lorentz-style transform (v12)

Within each 4D block, MonSTERs applies a structured Lorentz-style transform consisting of:

1. a **boost** (hyperbolic rotation) mixing the time coordinate with one spatial component using (\cosh\phi) and (\sinh\phi), and
2. a **spatial rotation** in an orthogonal spatial plane using (\cos\theta) and (\sin\theta).

This is implemented in the same computational spirit as RoPE: closed-form scalar updates over small blocks rather than explicit dense matrix multiplications.

The important point for correctness is not the particular choice of planes, but that each block transform is an element of (SO(1,3)) (or an isometry of (\eta)) and that its parameters depend linearly on (s).

---

## 5. The MonSTERs fusion identity (main result)

This section is the formal analogue of RoPE’s core guarantee.

### 5.1 Statement

Let (L(s)) be the MonSTERs absolute transform for position (s=(t,x,y,z)). Define transformed query and key blocks:
[
q' = L(s_q),q,
\qquad
k' = L(s_k),k,
]
and measure similarity via the Minkowski inner product:
[
\langle q',k'\rangle_\eta = q'^\top \eta k'.
]

**Theorem (absolute→relative fusion in Minkowski space).**
If (i) (L(s)) preserves the Minkowski metric and (ii) the family ({L(s)}) composes additively with respect to (s), then
[
\boxed{
\langle L(s_q)q,; L(s_k)k\rangle_\eta
=====================================

\langle q,; L(s_k-s_q)k\rangle_\eta.
}
]

### 5.2 Proof

Start from the definition:
[
\langle L(s_q)q,; L(s_k)k\rangle_\eta
=====================================

# (L(s_q)q)^\top \eta (L(s_k)k)

q^\top L(s_q)^\top \eta L(s_k)k.
]

Because (L(s)) is an isometry of (\eta),
[
L(s)^\top \eta L(s) = \eta
\quad\Rightarrow\quad
L(s)^\top \eta = \eta L(s)^{-1}.
]
Substituting:
[
q^\top L(s_q)^\top \eta L(s_k)k
===============================

q^\top \eta L(s_q)^{-1}L(s_k)k.
]

If the transforms compose additively in the position parameter (the direct analogue of RoPE’s (R(m)^{-1}R(n)=R(n-m))),
[
L(s_q)^{-1}L(s_k)=L(s_k-s_q),
]
so
[
q^\top \eta L(s_q)^{-1}L(s_k)k
==============================

# q^\top \eta L(s_k-s_q)k

\langle q,; L(s_k-s_q)k\rangle_\eta.
]
This completes the proof. ∎

### 5.3 Why the additive composition condition holds here

In RoPE, additivity holds because the rotation angle is linear in position and rotations form a group. MonSTERs mirrors this: rapidity and rotation angles are linear in spacetime coordinates, and boosts/rotations are elements of a group of Minkowski isometries. Under this construction, subtracting positions corresponds to composing inverse/forward transforms.

Practically, this is the check that matters for implementation: applying absolute transforms to (q) and (k) should be equivalent (in the similarity computation) to applying the relative transform associated with (s_k-s_q) to one side.

---

## 6. Why MonSTERs resolves axial RoPE’s diagonal weakness

Axial RoPE builds multi-dimensional position from independent axis actions, so there is no primitive channel whose phase depends jointly on multiple coordinates. MonSTERs instead uses a single coupled parameterization:
[
(\phi,\theta^x,\theta^y,\theta^z) \propto (t,x,y,z),
]
so the induced relative transform (L(s_k-s_q)) is inherently sensitive to general directions in spacetime. Mixed-direction effects—diagonals in 2D, or arbitrary spatiotemporal directions in 4D—are not deferred to learned mixing after attention; they are present in the positional mechanism itself.

---

## 7. Implementation notes as a specification (v12)

This section is meant to be directly actionable when writing or refactoring code.

1. **Positions** are 4-vectors (s=(t,x,y,z)).
2. **Embedding partition**: split channels into frequency buckets; each bucket is a triad of 4D blocks (12 channels).
3. **Frequency schedule**: assign each bucket an inverse frequency (\lambda_j) (typically geometric).
4. **Parameterization**: compute (\phi_j,\theta^x_j,\theta^y_j,\theta^z_j) as linear functions of (s).
5. **Transform**: apply closed-form boost + spatial rotation updates per 4D block.
6. **Similarity**: use a Minkowski inner product per block.
7. **Correctness tests** (non-negotiable):

   * **Metric preservation**: verify (\langle L(s)v,;L(s)v\rangle_\eta = \langle v,v\rangle_\eta) numerically per block.
   * **Fusion identity**: verify (\langle L(s_q)q,;L(s_k)k\rangle_\eta \approx \langle q,;L(s_k-s_q)k\rangle_\eta).

These two checks correspond exactly to “isometry + additivity,” which are the minimal mathematical conditions behind the RoPE-style fusion proof.

---

## 8. Summary

MonSTERs is a direct continuation of the RoPE trajectory:

* RoPE: SO(2) rotations + Euclidean dot product → absolute transforms yielding relative dependence.
* MonSTERs: SO(1,3) Lorentz-style transforms + Minkowski inner product → the same absolute→relative fusion in a 4D spacetime setting.

In contrast to axial RoPE, which enforces axis-separable structure, MonSTERs encodes coupled spatiotemporal relationships intrinsically while retaining RoPE’s modification-light and drop-in integration style.