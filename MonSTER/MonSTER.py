"""
MonSTER: Minkowski Space-Time Embedding Rotors

This module provides functions to compute MonSTER, a 4D generalization of
RoPE (Rotary Position Embedding). It calculates a unique 4D Lorentz
transformation, R_eff, based on the relative spacetime displacement
between query and key elements, allowing a transformer's attention
mechanism to become explicitly aware of spacetime geometry.

The core of the process involves two key stages:
1.  **Unit-Normalization**: Physical spacetime displacements (Δt, Δx, Δy, Δz)
    are converted into dimensionless "lattice units" where the speed of
    light c=1. This is achieved by defining a characteristic time-step
    `tau = s / c` from a chosen spatial scale `s`. This principled
    normalization avoids numerical instability in the subsequent hyperbolic
    calculations without needing ad-hoc clamping.
2.  **Rotor Generation**: The normalized displacements are scaled by different
    inverse frequencies for time and space across several blocks. For each
    block, a spatial rotation and a Lorentz boost are constructed from
    these scaled values and combined into a final 4x4 Lorentz transformation
    matrix, R_eff_b.

The resulting stack of R_eff matrices can then be used to modulate the
attention scores, ensuring that the same relative displacement always
produces the same geometric transformation.
"""

import jax
import jax.numpy as jnp

def get_monster_rotors(
    pos_q,
    pos_k,
    num_blocks: int,
    s: float = 1.0, # Spatial grid unit in meters, e.g., 2^k
    c: float = 299792458.0, # Speed of light in m/s
    base_time: float = 10000.,
    base_space: float = 10000.,
    epsilon: float = 1e-8,
    dtype=jnp.float32
):
    """Computes MonSTER rotors from query and key spacetime positions.

    Args:
        pos_q: Query positions (t, x, y, z). Shape (..., 4).
        pos_k: Key positions (t, x, y, z). Shape (..., 4).
        num_blocks: Number of frequency blocks (B) for multi-scale representation.
        s: The characteristic spatial grid spacing in physical units (e.g., meters).
           Choosing `s` as a power of two can be numerically advantageous.
        c: The speed of light in units consistent with `s` (e.g., m/s).
        base_time: The base for the geometric progression of temporal frequencies.
        base_space: The base for the geometric progression of spatial frequencies.
        epsilon: A small value for numerical stability when normalizing the rotation axis.
        dtype: The data type for all computations (e.g., jnp.float32).

    Returns:
        R_eff_blocks: A stack of 4x4 Lorentz transformation matrices, one for each
                      frequency block. The shape is (..., num_blocks, 4, 4).
    """
    pos_q = jnp.asarray(pos_q, dtype=dtype)
    pos_k = jnp.asarray(pos_k, dtype=dtype)

    # Step 1: Unit-Normalization (Lattice Units)
    # Define the time-step tau such that c = s/tau = 1 in lattice units.
    tau = s / c

    # Step 2: Raw Integer Displacement
    # Calculate displacement in physical units
    delta_pos_raw = pos_k - pos_q
    delta_t_raw = delta_pos_raw[..., 0]
    delta_coords_raw = delta_pos_raw[..., 1:]

    # Convert to normalized "lattice" displacements
    delta_n_t = delta_t_raw / tau
    delta_n_coords = delta_coords_raw / s

    # Compute rotors using the normalized displacements
    return _compute_rotors_from_normalized_displacements(
        delta_n_t=delta_n_t,
        delta_n_coords=delta_n_coords,
        num_blocks=num_blocks,
        base_time=base_time,
        base_space=base_space,
        epsilon=epsilon,
        dtype=dtype
    )


def _compute_rotors_from_normalized_displacements(
    delta_n_t,
    delta_n_coords,
    num_blocks: int,
    base_time: float,
    base_space: float,
    epsilon: float,
    dtype
):
    """
    Computes MonSTER rotors from normalized spacetime displacements (in lattice units).
    This is an internal helper function.
    """
    # Step 3: Frequency Scaling
    freqs = jnp.arange(num_blocks, dtype=dtype)
    inv_freq_time = 1.0 / (base_time ** (freqs / num_blocks))
    inv_freq_space = 1.0 / (base_space ** (freqs / num_blocks))

    delta_t_scaled = jnp.einsum('...,b->...b', delta_n_t, inv_freq_time)
    delta_s_scaled = jnp.einsum('...i,b->...bi', delta_n_coords, inv_freq_space)

    # Step 4: Compute Boost Rapidity
    phi_b = delta_t_scaled

    # Step 5: Compute Spatial Rotation
    theta_b = jnp.linalg.norm(delta_s_scaled, axis=-1, ord=2)

    default_spatial_axis = jnp.array([0., 0., 1.], dtype=dtype)
    # Ensure default axis shape matches for broadcasting
    axis_shape = delta_s_scaled.shape
    default_axis_bc = jnp.broadcast_to(default_spatial_axis, axis_shape)

    is_zero_spatial_delta = theta_b < epsilon
    axis_u_rot_b = jnp.where(
        is_zero_spatial_delta[..., None],
        default_axis_bc,
        delta_s_scaled / jnp.maximum(theta_b[..., None], epsilon)
    )

    # Step 6: Build Block-wise Transforms (Rotation)
    R3_b = _build_rotation_matrix(axis_u_rot_b, theta_b)

    pref_B_shape = R3_b.shape[:-2]
    M_rot_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_rot_b = M_rot_b.at[..., 0, 0].set(1.0)
    M_rot_b = M_rot_b.at[..., 1:, 1:].set(R3_b)

    # Step 6: Build Block-wise Transforms (Boost)
    ch_b = jnp.cosh(phi_b)
    sh_b = jnp.sinh(phi_b)
    # The boost and rotation axes are the same
    axis_u_boost_b = axis_u_rot_b

    M_boost_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_boost_b = M_boost_b.at[..., 0, 0].set(ch_b)
    M_boost_b = M_boost_b.at[..., 0, 1:].set(-axis_u_boost_b * sh_b[..., None])
    M_boost_b = M_boost_b.at[..., 1:, 0].set(-axis_u_boost_b * sh_b[..., None])

    eye3 = jnp.eye(3, dtype=dtype)
    uuT_boost_b = jnp.einsum('...bi,...bj->...bij', axis_u_boost_b, axis_u_boost_b)
    ch_b_minus_1_exp = (ch_b - 1.0)[..., None, None]

    M_boost_b = M_boost_b.at[..., 1:, 1:].set(eye3 + ch_b_minus_1_exp * uuT_boost_b)

    # Step 7: Combine into the Effective Rotor
    # R_eff = M_boost @ M_rot. Since axes are shared, they commute.
    R_eff_blocks = jnp.einsum("...bij,...bjk->...bik", M_boost_b, M_rot_b)

    return R_eff_blocks


def _build_rotation_matrix(axis, theta):
    """
    Rodrigues' formula for 3x3 rotation about 'axis' by angle 'theta'.
    Handles broadcasting for batched axes (..., B, 3) and angles (..., B).
    """
    theta_exp = theta[..., None]
    cos_t = jnp.cos(theta_exp)
    sin_t = jnp.sin(theta_exp)

    uuT = jnp.einsum('...bi,...bj->...bij', axis, axis)

    zeros = jnp.zeros_like(axis[..., 0])
    u_cross = jnp.stack([
        zeros, -axis[..., 2], axis[..., 1],
        axis[..., 2], zeros, -axis[..., 0],
        -axis[..., 1], axis[..., 0], zeros
    ], axis=-1).reshape((*axis.shape[:-1], 3, 3))

    I3 = jnp.eye(3, dtype=axis.dtype)
    cos_t_exp_mat = cos_t[..., None]
    sin_t_exp_mat = sin_t[..., None]

    return (cos_t_exp_mat * I3 +
            (1 - cos_t_exp_mat) * uuT +
            sin_t_exp_mat * u_cross)