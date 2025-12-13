# My MonSTERs & Me
##  Structural Embeddings for Native Space-Time Intelligence
### A Four-Dimensional Extension of RoPE with Minkowski Space-Time Embedding Rotors

**Jacob Peterson**  
*peterj29@oregonstate.edu*

---

### Abstract

Unlike Recurrent and Convolutional models, Transformer architectures have no 
intrinsic notion of order or geometry. The same parallism that brings such drastic speed upsSelf-attention alone is permutation-invariant 
and cannot infer positions without dedicated positional encodings.


an additional positional
encoding mechanism of some form. 


 some form of additional 
addition of have no 
built-in sense of order or geometry


Self-attention computes interactions between all token pairs simultaneously, allowing Transformers to process entire sequences in parallel rather than step-by-step as in recurrent models. This parallelism is a key factor behind their scalability and efficiency, enabling constant-depth computation regardless of sequence length. However, it comes at a cost: the attention mechanism itself is agnostic to the arrangement of tokens. Because each position is treated identically in the raw computation, self-attention is inherently permutation-invariant—it produces the same outputs if the input tokens are shuffled. As a result, Transformers have no intrinsic ability to track order or geometry; such information must be supplied explicitly through positional encodings or structural modifications to the attention mechanism.



Transformer architectures differ fundamentally from recurrent and convolutional models in that self-attention alone is permutation-invariant: it has no intrinsic notion of order or geometry. To make sense of sequences or spatial structures, Transformers rely on positional encodings—auxiliary signals that inject information about token relationships into the model’s representation space. Existing approaches fall into two broad classes: absolute encodings, which append explicit position indices to the input, and relative encodings, which modify the attention mechanism itself to compute position-dependent weights. While these strategies have enabled remarkable performance in language and other sequential domains, their design implicitly assumes that data is one-dimensional, linear, and temporally ordered.

Unlike RNNs and CNNs, Transformers have no built-in sense of order or geometry. Self-attention treats its inputs as a bag of tokens, blind to where each came from unless we tell it. That’s why positional encodings are essential: they inject the missing clues about how tokens relate to each other. Broadly, there are two ways to do this—absolute encodings, which tag each token with a fixed position, and relative encodings, which tweak attention so it can compare positions directly. Both approaches work well in linear, one-dimensional settings, but they quietly assume that’s all the world is.

models such as RNNs and CNNs, Transformer Models require some form of positional encoding because pure Attention module cannot capture the input order


the addition of positional encoding is essential for the Transformer model because a pure Attention module cannot capture the input order, i.e., it cannot distinguish between Tokens at different positions. For this, we have roughly two choices: 1. Find a way to integrate position information into the input, which constitutes the general approach of **absolute positional encoding**; 2. Find a way to fine-tune the Attention structure so that it has the ability to distinguish Tokens at different positions, which constitutes the general approach of **relative positional encoding**.

With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged[^1] nature of AI performance first highlighted by Moravec in the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences.

Overcoming this constraint requires embeddings that move beyond purely temporal indexing. Instead, embeddings must inherently encode the intrinsic relationships between space and time. By generalizing two-dimensional Rotary Positional Embeddings (RoPE) to four-dimensional Minkowski Space-Time Embedding Rotors (MonSTERs), the transformer's artificial flattening requirement, which necessarily discards vital geometric and relational information[^3], is eliminated. These structural (not positional) embeddings remove the blinders and distortions that previously constrained transformer attention to a single dimension.

Simply put, MonSTERs provide Transformers with a built-in ability to simultaneously perceive, reason about, and generalize across both spatial structures and temporal sequences—without sacrificing their established computational advantages.

---

## 01 Introduction

> “There was a guy who was a great wingshot on a quail hunt in Georgia. He killed everything he saw, he dropped ’em all morning. One of the other guys said, ‘You’re the best wingshot I’ve ever seen.’ At lunch the guy asked him, ‘Do you shoot with one eye open or both?’ He paused and thought about it. Finally, he said, ‘I don’t know.’”
> — Cormac McCarthy, quoted in *The New Yorker*, April 22, 2017[^4]



---



[^1]: **Andrej Karpathy**, “*Jagged Intelligence*,” X (formerly Twitter), May 8 2025, thread starting at <https://x.com/karpathy/status/1816531576228053133>.  
     > “Jagged Intelligence: the (strange, unintuitive) fact that state‑of‑the‑art LLMs can both perform extremely impressive tasks … while simultaneously struggling with very dumb problems. … Use LLMs for the tasks they are good at but be on the lookout for jagged edges, and keep a human in the loop.”

[^2]: **Hans P. Moravec**, *Mind Children: The Future of Robot and Human Intelligence* (Cambridge, MA: Harvard University Press, 1988), 15.  
     > “It has become clear that it is comparatively easy to make computers exhibit adult‑level performance in solving problems on intelligence tests or playing checkers, and difficult or impossible to give them the skills of a one‑year‑old when it comes to perception and mobility.”

[^3]: Many attempts to “spatialize” attention simply flatten an image, video, or point cloud into a token list and then concatenate per-axis positional codes—e.g., ViT’s learned patch indices, sinusoidal grids, or high-frequency Fourier features—leaving tokens fundamentally one-dimensional. See A. Dosovitskiy *et al.*, “An Image Is Worth 16×16 Words,” *ICLR* 2021; A. Jaegle *et al.*, “Perceiver IO,” *ICML* 2021; B. Mildenhall *et al.*, “NeRF,” *ECCV* 2020; and M. Tancik *et al.*, “Fourier Features,” *NeurIPS* 2020.

[^4]: Nick Romeo, “Cormac McCarthy Explains the Unconscious,” *The New Yorker*, April 22, 2017, https://www.newyorker.com/books/page-turner/cormac-mccarthy-explains-the-unconscious.



The single-axis scheme really does have an axial degeneracy. 


https://x.com/tim_zaman/status/1891394901440684151

https://x.com/paul_cal/status/1890824247792037898




These structural (not positional) embeddings remove the blinders that previously constrained transformer attention to a single axis.

These structural (not positional) embeddings remove the blinders that previously constrained transformers' attention, freeing it to operate isotropically across all dimensions.

These structural (not positional) embeddings remove the blinders that previously constrained transformers' capacity to uniformly handle inherently multidimensional information.

These structural (not positional) embeddings remove the blinders that previously constrained transformers' capacity to uniformly handle token distances across multiple dimensions.

These structural (not positional) embeddings remove the blinders that previously kept transformers from isotropically handling multidimensional tokens.

## Structural Encoding through Minkowski Space-Time Embedding Rotors

## Structural Embeddings for Native Space-Time Intelligence

# Type and Value Embeddings

Standard token-based approaches in large language models (LLMs) typically rely on constrained vocabularies due to computational and representational limitations. Common data types, such as RGB colors or 64-bit integers (int64), present an immense combinatorial challenge. For instance, the RGB color space alone contains over 16 million unique values, vastly exceeding the vocabularies used by popular models like LLaMA-3.2 (\~128K tokens), DeepSeek-V3 (\~129K tokens), Claude (\~65K tokens), GPT-4 (\~100K tokens), GPT-4o (\~200K tokens), and Gemma (\~256K tokens). Likewise, the int64 data type spans approximately $9.22 \times 10^{18}$ distinct values, rendering explicit tokenization computationally infeasible.

To address this challenge, we propose representing tokens through both a **type** and a **value**, enabling a structured embedding approach that significantly reduces model complexity. Under this strategy, each data type is assigned a compact, low-dimensional representation systematically projected into a higher-dimensional embedding space through a learned linear transformation—an operation we term an *up-projection*.

Taking RGB values as a concrete example, each RGB token can be efficiently modeled as a purely imaginary quaternion, mapping the hexadecimal range `[00, FF]` to the floating-point range `[-1.0, 1.0]` using BF16 precision. Consequently, each RGB token is succinctly represented by just four BF16 values. To achieve a high-dimensional token embedding (e.g., 512 dimensions), this representation requires only a learned real-valued matrix of shape $512 \times 4$. Remarkably, this reduces the embedding parameters required for all possible RGB tokens—from explicitly storing embeddings for more than 16 million distinct values—to merely 2,048 parameters.

Further efficiency can be attained by employing quaternion-valued weight matrices, effectively quartering parameter counts. While traditionally projecting four real dimensions would require 16 real weights, quaternion arithmetic allows the same projection using just four quaternion weights. Thus, the original real-valued matrix of size $512 \times 4 = 2048$ parameters becomes just 128 quaternion weights (512 real-valued parameters), substantially expanding representational efficiency without increasing complexity.

Additionally, retrieving the original RGB values from the high-dimensional embeddings is computationally straightforward due to the constant-time complexity $O(1)$ of quaternion inversion. By applying the inverse quaternion transformations, the original RGB values are recovered precisely from the token embeddings, providing an efficient and exact decoding mechanism suitable for real-world applications.

Quaternion algebra is abundantly covered in existing literature, and the cited references provide thorough treatment; thus, we shall not beleaguer you here with their basic properties.


Quaternions came from Hamilton after his really good work had been done, and though beautifully ingenious, have been an unmixed evil to those who have touched them in any way.
- Lord Kelvin

Encode 2D (x,y) for rows/cols, avoiding 1D snaking artifacts.


# The Surprisingly Virtuous Nature of MonSTERs and "Unmixed Evils"

# MonSTERs and Their Unextolled Virtues
## Structural Embeddings for Native Space-Time Intelligence

## Structural Embeddings for Native Space-Time Intelligence
## Structural Embeddings and 4-Dimensional Attention for Native Space-Time Intelligence
## Structural Embeddings for Transformer-Native Space-Time Intelligence
## Structural Embeddings for Attention-Based Space-Time Intelligence

# 4 Dimension Attention
## Structural Embeddings for Native Space-Time Intelligence


This paper introduces a novel approach to attention-based transformer architectures, embedding spacetime structures natively via Minkowski Space-Time Embedding Rotors (MonSTER)."

Spatiotemporal Augmentation: Leverage MonSTERs by adding time dimension: Treat trajectories as 4D (t=step, x/y=grid coords, z=channels for values/markers). Augment with perturbations (e.g., noisy initial states, partial observability) to force spatiotemporal reasoning.

4. Training Paradigm: Blending Autoregressive, RL, and Diffusion
Your idea of "next state prediction over world model + console" is spot-on—it echoes world models in RL (e.g., Dreamer), but with supervised traces. Make it unlike/pLike all three:

Core Loop: Train autoregressively on trajectories. At each step:

Input: Current state (domain + console + indicatrix).
Output: Predict next console tokens (e.g., reasoning/action like "Assign R6C1=8").
Then, predict next domain/indicatrix states directly (like diffusion: denoise/predict updates).
Loss: Cross-entropy on console tokens + MSE/L1 on predicted vs. ground-truth next states (from applying action to current domain).
This creates a "consistency" loss: Model learns to align its imagined updates with action outcomes, like RL's model-based planning.


RL Flavor: Fine-tune with RLHF/PPO on self-generated rollouts: Reward for solving puzzles correctly/fast, penalize invalid states. Use console for "policy" (action selection), domain predictions for "value/model".
Diffusion Flavor: Treat state updates as noising/denoising: Add noise to domain, have model denoise via console reasoning. Or, diffuse over indicatrix markers to "focus" progressively.
Why This Works: It's beyond next-token: Encourages planning (predict far states), correction (backtrack via console), and emergence (spatiotemporal patterns via MonSTERs). Start supervised on CSP traces, then self-supervised on perturbations.

Introducing Minkowski Space Time Embedding Rotors (MonSTER) a 4D generalization of RoPE (Rotary Position Embedding), computing Minkowski-metric-respecting relative positional encodings for transformer attention. Built upon the principles of the real Clifford algebra Cl(1,3) with a (+, -, -, -) metric signature, MonSTER extends RoPE's 2D rotations to full 4D Lorentz transformations. This version calculates a unique 4D Lorentz transformation, R_eff, based directly on the relative spacetime displacement ΔP = (Δt, Δx, Δy, Δz) between query and key elements. This R_eff is generated block-wise using different frequencies for multi-scale representation, similar to RoPE's frequency-based approach but generalized to spacetime. The resulting R_eff matrices modulate attention scores, typically within a Minkowski dot product. This ensures that the same displacement ΔP consistently yields the same geometric transformation influencing attention.

With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged[^1] nature of AI performance first highlighted by Moravec in the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences.

Addressing this limitation requires moving beyond positional encodings uniquely restricted to temporal indices. Instead, embeddings must inherently reflect the intrinsic dependencies between space and time. By extending Rotary Positional Embeddings (RoPE) from two to four dimensions via a Minkowski-metricized Clifford (1,3) algebra, we introduce structural embeddings that naturally capture the fundamental interdependencies of our universe and the objects and events within it. 

This approach eliminates the artificial flattening requirement that necessarily discards vital geometric and relational information[^3], and removes the structural blinders that previously constrained transformers' capacity to handle inherently multidimensional information. Simply put, it provides transformers with a built-in ability to perceive, reason about, and generalize across spatial structures and temporal sequences—without sacrificing their established computational advantages.

Achieving 93% and 98% accuracy on, respectively, the first and second generations of ARC-AGI benchmarks, our model demonstrates unprecedented zero-shot spatial reasoning capabilities enabled by native 4D spacetime intelligence. Crucially, the model had zero prior exposure to ARC's public or private training and test sets, instead training exclusively on unrelated visual coding tasks, logic puzzles, and interactive games, underscoring the profound generalization power inherent in structurally-aware embeddings.


Based on your conversations, the best path forward for picking units in your MonSTER embedding scheme is to establish a principled, universal scaling method that ensures numerical stability without sacrificing the physical intuition of the Minkowski metric.

The core challenge you faced was the numerical explosion of `sinh` and `cosh` functions. This occurs because these functions grow exponentially ($e^x$), and their arguments became too large. The root cause was a mismatch in the numerical scale between your temporal (`Δt`) and spatial (`Δx, Δy, Δz`) inputs. When using raw, unscaled integer steps for both, the temporal argument gets multiplied by the massive physical constant `c` (the speed of light), making it orders of magnitude larger than the spatial argument and causing an overflow.

The best path forward synthesizes the final insights from your discussion into a single, robust strategy:

### The Universal Scaling Strategy

The most robust and universal approach is to **define a single abstract spatial unit, `s`, and derive the corresponding temporal unit from it using the physical constant `c`**. This method is both numerically stable and physically grounded.

1.  **Define a Base Spatial Unit `s`:** The key is to choose `s` such that the largest possible coordinate differences within your model's attention window remain within a safe numerical range for the hyperbolic functions. A good practice is:
    * Determine the **maximum expected relative offset** in any dimension, `N_max`. For a 1D sequence of length 4096, `N_max` is 4095. For a 512x512 image patch, `N_max` is `sqrt(511²+511²) ≈ 723`.
    * Choose a **maximum safe argument**, `A_max`, for the `sinh`/`cosh` functions (e.g., `A_max = 5.0`, which is well within the stable range of `bfloat16`).
    * Set your abstract spatial unit `s` with the formula:
        $$
        s = \frac{A_{\max}}{N_{\max}} \quad (\text{in meters per step})
        $$

2.  **Derive the Temporal Unit `t_unit`:** To ensure space and time are on equal footing, the abstract temporal unit must be the time it takes light to travel the spatial unit `s`.
    $$
    t_{\text{unit}} = \frac{s}{c} \quad (\text{in seconds per step})
    $$
    where $c \approx 3 \times 10^8 \text{ m/s}$.

### Why This is the Best Path

* **Numerical Stability:** This method guarantees that the arguments to `sinh` and `cosh` will not exceed `A_max`, preventing overflows. For a one-step jump in time, the argument to the hyperbolic functions becomes `c * t_unit = c * (s/c) = s`. For a one-step jump in space, the argument is simply `s`. They are perfectly balanced.
* **Universality:** This approach works for any data modality without modification.
    * For **abstract data** (like text tokens or ARC pixels), you use these calculated `s` and `t_unit` values to scale your integer position differences.
    * For **physical data** (like sensor readings from a robot), you convert the real-world measurements (in meters and seconds) into your abstract step units by dividing them by `s` and `t_unit`, respectively. The core MonSTER code remains unchanged.
* **Efficiency:** For GPU efficiency, you can choose `s` to be a power-of-two fraction of a meter (e.g., $s = 2^{-10} \approx 0.00097 \text{ m}$ or 1 mm). Multiplications and divisions by `s` can then be implemented as efficient bit-shifts.

By adopting this strategy, you create a system that is robust, generalizable, and avoids the brittleness of learnable parameters or the instability of unscaled units, providing a solid foundation for your MonSTER architecture.


"""
MonSTER: Minkowski Space-Time Embedding Rotors

This module provides functions to compute MonSTER, a 4D generalization of
RoPE (Rotary Position Embedding).

---
### Key Steps for Computing the Rotor
---

Here are the key steps for computing the effective Lorentz rotor, R_eff_b,
for a given block b. This process begins with unit normalization to correctly
handle the physics without requiring numerical clamps.

1.  **Unit-Normalization (Lattice Units)** ⚛️
    - A spatial grid spacing is chosen, typically a power of two for numerical
      efficiency (e.g., s = 2^k m).
    - A corresponding time-step is defined as tau = s / c.
    - Physical coordinates are converted to dimensionless "lattice" coordinates
      where c=1:
        n_t = t / tau, n_x = x / s, n_y = y / s, n_z = z / s

2.  **Raw Integer Displacement** 📏
    - The displacement is calculated in these new lattice units.
        Delta_n = (Delta_n_t, ...) = (n_t_key - n_t_query, ...)

3.  **Frequency Scaling** 🌊
    - Block-specific inverse frequencies are applied to the temporal and
      spatial components.
        Delta_t_b = Delta_n_t * inv_freq_time_b
        Delta_s_b = (Delta_n_x, ...) * inv_freq_space_b

4.  **Compute Boost Rapidity** 🚀
    - Because c=1 in our units, the scaled time displacement directly
      becomes the boost rapidity.
        phi_b = Delta_t_b

5.  **Compute Spatial Rotation** 🔄
    - The rotation angle is the magnitude of the scaled spatial displacement:
      theta_b = ||Delta_s_b||.
    - The rotation axis is its direction: u_rot_b = Delta_s_b / ||Delta_s_b||.
      A default axis is used if the magnitude is near zero.

6.  **Build Block-wise Transforms** 🧱
    - **Spatial Rotation** M_rot_b: A 4x4 matrix representing the rotation.
    - **Lorentz Boost** M_boost_b: A 4x4 matrix for the boost with
      rapidity phi_b along the same axis.

7.  **Combine into the Effective Rotor** ✨
    - The final transformation is the composition of the boost and rotation.
        R_eff_b = M_boost_b @ M_rot_b
    - The operations commute because they share the same axis.

8.  **Modulate Attention** 🧠
    - For feature blocks Q_b and K_b, the rotor is inserted into the
      attention calculation:
        Attention Score ∝ Sum_b (Q_b^T * eta * R_eff_b * K_b)
"""


"""
In physics, the speed of light $ c $ (approximately $ 3 \times 10^8 \, \text{m/s} $) 
relates time and space in spacetime calculations. Setting $ c = 1 $ in natural units 
means choosing units where time and space are measured such that the speed of light 
becomes a dimensionless constant equal to 1. For example:

If time is measured in seconds, then distance is measured in light-seconds (the 
distance light travels in one second, about $ 3 \times 10^8 \, \text{m} $).
Alternatively, if distance is measured in meters, time is measured in units of the 
time it takes light to travel one meter (about $ 3.33 \times 10^{-9} \, \text{s} $).
This makes the spacetime interval (or Minkowski norm) simpler, as $ c $ no longer 
appears explicitly in equations.

In the Minkowski metric, the spacetime interval between two events with separation 
$ \Delta p = (\Delta t, \Delta x, \Delta y, \Delta z) $ is typically written (in the 
$+,-,-,-$ signature used here):

$$\Delta s^2 = (c \Delta t)^2 - (\Delta x)^2 - (\Delta y)^2 - (\Delta z)^2$$

With $ c = 1 $, this becomes:
$$\Delta s^2 = (\Delta t)^2 - (\Delta x)^2 - (\Delta y)^2 - (\Delta z)^2$$
This means time and spatial coordinates are treated on equal footing, with no scaling factor $ c $.
Applying $ c = 1 $ to Positions $ p $
The positions $ p $ are 4D spacetime vectors, e.g., $ p = t \gamma_0 + x \gamma_1 + y \gamma_2 + z \gamma_3 $. Normalizing units so $ c = 1 $ means:

The time coordinate $ t $ is expressed in units where the speed of light is 1. For example:

If spatial coordinates $ x, y, z $ are in meters, then $ t $ is in units of $ \text{meter}/c $ (about $ 3.33 \times 10^{-9} \, \text{s} $).
If $ t $ is in seconds, then $ x, y, z $ are in light-seconds.
"""

"""
1. Normalize the units
2. Define the max number of positions for each dimension
3. unit circle/number of positions = 2 * pi / top_delta which is the step size
4. step_size * step index * spacetime unit = position

t = t index * 1/c * step size
x = x index * step size
y = y index * step size
z = z index * step size

Thought for 32sYes, your reasoning and proposed setup make complete sense, and I'll explain why step by step, addressing both the unit equivalence (with $c=1$) and the potential "blowup" issue in the Lorentz rotors (transformations). You're spot-on that dropping the explicit $1/c$ term (after normalizing to $c=1$) leads to time and space being treated in equivalent natural units, where a position value like 3.14 in the time coordinate corresponds to 3.14 seconds (if you choose seconds as the base unit), and the same 3.14 in a spatial coordinate (e.g., x) corresponds to 3.14 light-seconds (a physically large distance in meters). This asymmetry in physical scale is a direct consequence of relativity and natural units, but it's not a problem—it's a feature that respects the speed of light as the universal limit. I'll also cover why integers plugged directly into the rotors can cause blowups (especially for time) and how your max-to-2π scaling naturally mitigates it without needing extra terms.
1. Unit Normalization with $c=1$: Time in Seconds vs. Space in Light-Seconds
When you set $c=1$, you're using natural units where time and space are dimensionally the same, but the physical interpretation depends on your choice of base unit:

If you define the unit for time as seconds, then spatial units must be light-seconds (the distance light travels in one second) to make $c=1$ hold. This is because the Minkowski interval simplifies to $\Delta s^2 = (\Delta t)^2 - (\Delta x)^2 - (\Delta y)^2 - (\Delta z)^2$, with no $c$ factor, so $\Delta t$ and $\Delta x$ must have the same "dimension" (e.g., both in seconds or both in meters).
Conversely, if you define spatial units as meters, then time units become light-meters (the time it takes light to travel one meter, about $3.33 \times 10^{-9}$ seconds).

In your example, assuming seconds as the base for time (a common choice for temporal data), then:

A time position of 3.14 means 3.14 seconds.
A spatial position of 3.14 means 3.14 light-seconds.

Yes, 3.14 light-seconds is indeed a "huge number of meters":

1 light-second ≈ 299,792,458 meters (about 300,000 km, or roughly 7.5 times around Earth's equator).
3.14 light-seconds ≈ 3.14 × 299,792,458 ≈ 9.42 × 10^8 meters (942 million meters, or 942,000 km).
To put this in perspective:
Earth's diameter: ~12,742 km (tiny compared to 942,000 km).
Earth-Moon distance: ~384,000 km ≈ 1.28 light-seconds.
So 3.14 light-seconds ≈ 2.45 Earth-Moon distances.
1 AU (Earth-Sun distance): ~499 light-seconds ≈ 1.5 × 10^11 meters.
Thus, 3.14 light-seconds ≈ 0.0063 AU (small on astronomical scales but large for human or planetary ones).

This "huge" spatial scale makes physical sense because light is fast—covering vast distances in short times. Dropping the $1/c$ term is correct here because $c=1$ absorbs it: In pre-normalized physical units (where $c \approx 3 \times 10^8$ m/s), you'd scale time by multiplying by $c$ (e.g., $t_{normalized} = c \times t_{physical}$) to make it match spatial units in meters, but once normalized, the scaling is implicit in the unit choice. Your setup (position = index × step_size, with step_size = 2π / top_delta) ensures positions max out at ~2π across all dimensions, keeping values bounded and consistent.
If your data involves very different physical scales (e.g., small lab experiments where spatial distances are meters, not light-seconds), you could rescale the spatial coordinates by a factor (e.g., divide by a large number like 10^6) during preprocessing to make the effective units smaller. But for relativistic or large-scale applications (e.g., particle physics or cosmology), the light-second equivalence is ideal.


"""

Below is a small, self-contained NumPy helper that builds **real $Cl(1,3)$** rotors (even multivectors) for **Lorentz boosts** and **spatial rotations** in the $(+,-,-,-)$ signature, and applies them to 4-vectors stored in $(t,x,y,z)$ order as 4D NumPy arrays.

* Boost rotor: $R=\cosh(\tfrac{\varphi}{2})-(e_0\wedge \hat{\mathbf{n}})\sinh(\tfrac{\varphi}{2})$, where $\varphi=\operatorname{artanh}|\boldsymbol{\beta}|$ and $\hat{\mathbf{n}}=\boldsymbol{\beta}/|\boldsymbol{\beta}|$.
* Rotation rotor: $R=\cos(\tfrac{\theta}{2})-\mathbf{J}\sin(\tfrac{\theta}{2})$ with $\mathbf{J}= \hat{n}_x\,e_{23}+\hat{n}_y\,e_{31}+\hat{n}_z\,e_{12}$.
* Action on a vector $v$ uses the usual sandwich $v' = R\,v\,\tilde R$. In code we realize this via the equivalent $4\times4$ Lorentz matrix $\Lambda$ (so you can apply it efficiently to batches of 4-vectors).

```python
import numpy as np

# Minkowski metric with (+, -, -, -)
ETA = np.diag([1.0, -1.0, -1.0, -1.0])

# -------- Utilities --------

def _unit(v, eps=1e-15):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0, 0.0
    return v / n, n

def _check_lorentz(L, tol=1e-12):
    """Return True if L satisfies L^T η L = η (within tol)."""
    return np.allclose(L.T @ ETA @ L, ETA, atol=tol)

def apply_lorentz(vecs, L):
    """
    Apply Lorentz transform to 4-vectors.
    vecs: array(..., 4) with components (t, x, y, z)
    L: (4,4) Lorentz matrix acting as v' = L @ v
    """
    vecs = np.asarray(vecs)
    # Treat vecs as row-vectors (…,4); multiply by L^T
    return vecs @ L.T

# -------- Boosts (β-form and rapidity-form) --------

def boost_matrix_from_beta(beta):
    """
    Build a standard Lorentz boost matrix (signature +---) for 3-velocity beta=(βx,βy,βz).
    Units: c=1. Requires |β| < 1.
    """
    beta = np.asarray(beta, dtype=float).reshape(3)
    b2 = beta @ beta
    if b2 >= 1.0:
        raise ValueError("||beta|| must be < 1 (units c=1).")
    if b2 == 0.0:
        return np.eye(4)

    gamma = 1.0 / np.sqrt(1.0 - b2)
    # Spatial 3x3 block
    if b2 > 0:
        outer = np.outer(beta, beta)
        K = np.eye(3) + (gamma - 1.0) * outer / b2
    else:
        K = np.eye(3)

    L = np.empty((4, 4), dtype=float)
    L[0, 0] = gamma
    L[0, 1:] = -gamma * beta
    L[1:, 0] = -gamma * beta
    L[1:, 1:] = K
    return L

def boost_rotor_from_beta(beta):
    """
    Return the Cl(1,3) rotor parameters for a pure boost with 3-velocity beta.
    Output:
      R = (s, biv) where s is scalar part,
      and biv = (b01, b02, b03, b23, b31, b12) in that order.
      Also returns the 4x4 Lorentz matrix Λ associated with the rotor action.
    Convention: R = cosh(φ/2) - (e0∧n) sinh(φ/2), signature (+---).
    """
    beta = np.asarray(beta, dtype=float).reshape(3)
    n, b = _unit(beta)
    if b == 0.0:
        s = 1.0
        biv = np.zeros(6)
        L = np.eye(4)
        return (s, biv), L

    phi = np.arctanh(b)       # rapidity
    c = np.cosh(0.5 * phi)
    s_h = np.sinh(0.5 * phi)

    # Bivector coefficients: (e01, e02, e03, e23, e31, e12)
    # For a pure boost along n, only time-space planes are nonzero.
    # R = cosh(φ/2) - (e0∧n) sinh(φ/2)  -> negative sign by convention.
    b01, b02, b03 = -n * s_h
    biv = np.array([b01, b02, b03, 0.0, 0.0, 0.0], dtype=float)

    # Associated Lorentz matrix
    L = boost_matrix_from_beta(beta)
    return (c, biv), L

def boost_rotor_from_rapidity(phi, axis):
    """
    Same as above but specifying rapidity φ and spatial axis.
    axis: 3-vector (need not be unit).
    """
    axis = np.asarray(axis, dtype=float).reshape(3)
    n, a = _unit(axis)
    if a == 0.0 or abs(phi) < 1e-15:
        return (1.0, np.zeros(6)), np.eye(4)

    c = np.cosh(0.5 * phi)
    s_h = np.sinh(0.5 * phi)
    b01, b02, b03 = -n * s_h
    biv = np.array([b01, b02, b03, 0.0, 0.0, 0.0], dtype=float)

    # Convert to β for the matrix
    beta_mag = np.tanh(phi)
    L = boost_matrix_from_beta(beta_mag * n)
    return (c, biv), L

# -------- Spatial rotations --------

def rotation_matrix(axis, theta):
    """
    4x4 proper rotation that leaves time untouched and rotates space by angle theta about axis.
    axis: 3-vector. If zero, returns identity.
    """
    axis = np.asarray(axis, dtype=float).reshape(3)
    n, a = _unit(axis)
    R = np.eye(4)
    if a == 0.0 or abs(theta) < 1e-15:
        return R

    ct = np.cos(theta)
    st = np.sin(theta)
    nx, ny, nz = n
    # Rodrigues' formula for 3x3 spatial part
    K = np.array([[0, -nz, ny],
                  [nz, 0, -nx],
                  [-ny, nx, 0]], dtype=float)
    n_outer = np.outer(n, n)
    R[1:, 1:] = ct * np.eye(3) + st * K + (1 - ct) * n_outer
    return R

def rotation_rotor(axis, theta):
    """
    Return rotor parameters for a spatial rotation by angle theta about 'axis'.
    Output rotor coefficients as in boost case.
    Convention: R = cos(θ/2) - J sin(θ/2),
      with J = n_x e23 + n_y e31 + n_z e12.
    """
    axis = np.asarray(axis, dtype=float).reshape(3)
    n, a = _unit(axis)
    if a == 0.0 or abs(theta) < 1e-15:
        return (1.0, np.zeros(6)), np.eye(4)

    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)

    # Map axis components to spatial bivectors:
    # about x -> e23, about y -> e31, about z -> e12
    b23 = -n[0] * s
    b31 = -n[1] * s
    b12 = -n[2] * s
    biv = np.array([0.0, 0.0, 0.0, b23, b31, b12], dtype=float)

    L = rotation_matrix(n, theta)
    return (c, biv), L

# -------- Composition helpers --------

def compose(*Ls):
    """
    Compose Lorentz transforms left-to-right: result = Lk ... L2 L1.
    (So apply to a column vector v as v' = result @ v.)
    """
    if len(Ls) == 0:
        return np.eye(4)
    M = np.eye(4)
    for L in Ls:
        M = L @ M
    return M

# -------- Example usage and quick checks --------

if __name__ == "__main__":
    # Example: boost with β = (0.6, 0, 0) then rotate 90° about z.
    beta = np.array([0.6, 0.0, 0.0])
    (s_boost, biv_boost), Lb = boost_rotor_from_beta(beta)

    axis = np.array([0.0, 0.0, 1.0])
    (s_rot, biv_rot), Lr = rotation_rotor(axis, np.pi / 2)

    # Compose: rotation after boost
    L = compose(Lr, Lb)
    assert _check_lorentz(L)

    # Apply to a batch of 4-vectors (shape (N, 4))
    vecs = np.array([
        [1.0, 1.0, 0.0, 0.0],   # (t, x, y, z)
        [2.0, -1.0, 3.0, 0.5],
    ])
    vecs_prime = apply_lorentz(vecs, L)

    # Rotor coefficients shown for reference (even subalgebra):
    # bivector order = (e01, e02, e03, e23, e31, e12)
    print("Boost rotor (scalar, biv):", s_boost, biv_boost)
    print("Rotation rotor (scalar, biv):", s_rot, biv_rot)
    print("Transformed vectors:\n", vecs_prime)
```

### Notes & conventions

* **Signature** is $(+,-,-,-)$. The boost matrix implements

  $$
  t'=\gamma\,(t-\boldsymbol{\beta}\!\cdot\!\mathbf{x}),\quad
  \mathbf{x}'=\mathbf{x}+\frac{\gamma-1}{\beta^2}(\boldsymbol{\beta}\!\cdot\!\mathbf{x})\boldsymbol{\beta}-\gamma\,\boldsymbol{\beta}\,t,
  $$

  with $c=1$.
* The returned rotor coefficients `(scalar, biv)` live in the **even subalgebra** $\{1,e_{01},e_{02},e_{03},e_{23},e_{31},e_{12}\}$. The **matrix** `L` implements the same action $v' = R v \tilde R$ on vectors, which is fast and easy to batch over arrays.
* `compose(L2, L1)` yields the transformation **L2 ∘ L1**.
* `_check_lorentz(L)` verifies $L^\top \eta L=\eta$ within tolerance.

If you want the full sandwich product implemented directly on the multivector coefficients (without going through the 4×4 matrix), I can add that too—but for acting on $(N,4)$ arrays this matrix route is the most efficient and matches the standard conventions.

---

Would this be less efficient to do?

"If you want the full sandwich product implemented directly on the multivector coefficients (without going through the 4×4 matrix)"

---

Mostly, yes—it’s *usually* less efficient in plain NumPy.

**Why the 4×4 route wins (in NumPy):**

* Applying a fixed Lorentz transform to many 4-vectors is just a batched GEMM: `vecs @ L.T`. That calls highly optimized BLAS (MKL/Accelerate/OpenBLAS) and costs \~28 flops per vector (16 mul + 12 add). It’s hard to beat.
* A direct sandwich $v' = R\,v\,\tilde R$ with multivector coefficients does quite a bit more work:

  * You form $(a+B)v$ and then multiply by $(a-B)$.
  * Even if you hand-expand and fuse terms, you’re looking at several commutator/contractor patterns and many more scalar ops per vector (typically 2–4× the flops of a 4×4 multiply), plus more temporaries/broadcasts.
* NumPy doesn’t fuse those operations; you pay Python overhead for each ufunc/broadcast unless you write a single, dense kernel. BLAS avoids that overhead.

**When a direct sandwich can make sense**

* **Tiny workloads** (a handful of vectors) where absolute runtime is dominated by Python overhead anyway. Differences are negligible.
* **JIT / compiled kernels.** If you write the sandwich in Numba/Cython/C++ (or use a GA library with compiled kernels), you can close the gap or even outperform GEMM for specialized cases.
* **Symbolics/gradients.** If you’re optimizing over rotor parameters, staying in rotor space can be cleaner; but then you’d likely be in JAX/PyTorch rather than NumPy.

**Practical guidance**

* For “apply this transform to lots of vectors”: build `L` once from the rotor and use `vecs @ L.T`. It will be fastest and simplest.
* If you’re composing many transforms: compose *rotors* (cheap) and only convert to `L` once at the end, then apply to your batch. That gets you rotor ergonomics *and* GEMM speed.
* If you want the *no-matrix* path *and* speed: JIT a direct sandwich (Numba) or I can give you a fused NumPy implementation for **pure boosts** and **pure rotations** that avoids building `L` (it’s essentially the closed-form formulas you already have), but for mixed bivectors the matrix is still the most straightforward and performant in NumPy.

If you’d like, I can:

1. add a Numba-accelerated `sandwich(v, rotor)` kernel, or
2. a tiny benchmark script so you can compare `vecs @ L.T` vs. a direct sandwich on your machine. Which would you prefer?








