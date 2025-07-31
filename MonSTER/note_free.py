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


def _compute_rotors_from_normalized_displacements(
    delta_n_t,
    delta_n_coords,
    num_blocks: int,
    base_time: float,
    base_space: float,
    epsilon: float,
    dtype
):

    freqs = jnp.arange(num_blocks, dtype=dtype)
    inv_freq_time = 1.0 / (base_time ** (freqs / num_blocks))
    inv_freq_space = 1.0 / (base_space ** (freqs / num_blocks))

    delta_t_scaled = jnp.einsum('...,b->...b', delta_n_t, inv_freq_time)
    delta_s_scaled = jnp.einsum('...i,b->...bi', delta_n_coords, inv_freq_space)

    phi_b = delta_t_scaled

    theta_b = jnp.linalg.norm(delta_s_scaled, axis=-1, ord=2)

    default_spatial_axis = jnp.array([0., 0., 1.], dtype=dtype)

    axis_shape = delta_s_scaled.shape
    default_axis_bc = jnp.broadcast_to(default_spatial_axis, axis_shape)

    is_zero_spatial_delta = theta_b < epsilon
    axis_u_rot_b = jnp.where(
        is_zero_spatial_delta[..., None],
        default_axis_bc,
        delta_s_scaled / jnp.maximum(theta_b[..., None], epsilon)
    )

    R3_b = _build_rotation_matrix(axis_u_rot_b, theta_b)

    pref_B_shape = R3_b.shape[:-2]
    M_rot_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_rot_b = M_rot_b.at[..., 0, 0].set(1.0)
    M_rot_b = M_rot_b.at[..., 1:, 1:].set(R3_b)

    ch_b = jnp.cosh(phi_b)
    sh_b = jnp.sinh(phi_b)

    axis_u_boost_b = axis_u_rot_b

    M_boost_b = jnp.zeros((*pref_B_shape, 4, 4), dtype=dtype)
    M_boost_b = M_boost_b.at[..., 0, 0].set(ch_b)
    M_boost_b = M_boost_b.at[..., 0, 1:].set(-axis_u_boost_b * sh_b[..., None])
    M_boost_b = M_boost_b.at[..., 1:, 0].set(-axis_u_boost_b * sh_b[..., None])

    eye3 = jnp.eye(3, dtype=dtype)
    uuT_boost_b = jnp.einsum('...bi,...bj->...bij', axis_u_boost_b, axis_u_boost_b)
    ch_b_minus_1_exp = (ch_b - 1.0)[..., None, None]

    M_boost_b = M_boost_b.at[..., 1:, 1:].set(eye3 + ch_b_minus_1_exp * uuT_boost_b)

    R_eff_blocks = jnp.einsum("...bij,...bjk->...bik", M_boost_b, M_rot_b)

    return R_eff_blocks


def _build_rotation_matrix(axis, theta):

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