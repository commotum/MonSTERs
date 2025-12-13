# RoPE Demo - Proof

To show that a RoPE implementation works in all cases, you would need to demonstrate the following:

### 1. Equivalence of Formulations

    - Show that applying separate rotations to absolute positions yields the same 
      result as the complex relative position formula. For any 2D vectors q and k 
      at positions m and n:

    - np.dot(apply_rotation_2d(q, m), apply_rotation_2d(k, n)) 
      must equal complex_dot_product_2d(q, k, m-n).

### 2. Norm Preservation

    - Show that the rotation doesn't change the vector's length.
    - The magnitude of a vector should remain constant after applying 
      the positional encoding.

    - np.linalg.norm(q) must equal np.linalg.norm(apply_rotation_2d(q, m)).

### 3. Shift Invariance

    - Show that the dot product is the same for any pair of tokens that have
      the same relative distance, regardless of where they are in the sequence.
      
    - For any shift d:
      np.dot(apply_rotation_2d(q, m), apply_rotation_2d(k, n)) must equal 
      np.dot(apply_rotation_2d(q, m+d), apply_rotation_2d(k, n+d)).

### 4. Extensibility to N-Dimensional Vectors

    - Show that by applying rotations blockwise to pairs of dimensions RoPE 
      naturally extends to higher-dimensional embeddings, interpreting the 
      embedding vector as complex pairs:

    - For an embedding vector $\mathbf{q} = (q_1, q_2, q_3, q_4, \dots, q_d)$, 
      treat it as complex-valued pairs:
      $\mathbf{q} = (q_1 + i q_2,\; q_3 + i q_4,\; \dots,\; q_{d-1} + i q_d) \in \mathbb{C}^{d/2}$.

### 5. Multiscale Coverage and Remote Attenuation

    - Show that by using the same geometric frequency schedule as the original 
      sinusoidal positional encodings, $\theta_i = 10000^{-2i/d}$ we ensure:
    - Multiscale Coverage: Frequencies span multiple positional scales.
    - Remote Attenuation: Increasing positional distances naturally attenuate 
      similarity due to phase misalignment, enhancing the model’s locality bias.