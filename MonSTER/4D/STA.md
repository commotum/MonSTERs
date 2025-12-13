To extend Rotary Position Embedding (RoPE) to 4-dimensional spacetime using the geometric algebra 
Spacetime Algebra, or STA, which is the Clifford algebra Cl(1,3) with Minkowski signature
we can leverage the algebraic structure to generalize the rotational encoding while incorporating 
the indefinite metric, light cones, and causality. This builds on RoPE's core idea—encoding 
absolute positions such that inner products capture relative positions—but adapts it to positions 
that are 4D spacetime vectors $ p = t \gamma_0 + x \gamma_1 + y \gamma_2 + z \gamma_3 $, where the 
light cone separates timelike (causal), spacelike (non-causal), and null (lightlike) intervals.

RoPE works in high-dimensional embedding spaces by grouping dimensions into 2D planes and applying 
rotations parameterized by a 1D position $ m $, achieving relative dependence via 
$ \mathbf{q}^T R_{m-n} \mathbf{k} $ and remote attenuation for long distances. For 4D spacetime 
positions, we'll group embeddings into 4D blocks. and use Lorentz transformations (rotations and 
boosts in STA) instead of pure Euclidean rotations. This preserves the Minkowski norm in each block, 
allowing the light cone to "enforce causality" by passing a relativity respecting signal.

Represent Embeddings and Positions in STA:

Treating each 4D block of the embedding vector (e.g., for queries $ \mathbf{q} $ or keys $ \mathbf{k} $) 
as a spacetime vector: $ v = v^0 \gamma_0 + v^1 \gamma_1 + v^2 \gamma_2 + v^3 \gamma_3 $, 
$ \gamma_0^2 = +1 $ (timelike), $ \gamma_i^2 = -1 $ (spacelike for $ i=1,2,3 $). The squared norm 
$ v^2 = (v^0)^2 - (v^1)^2 - (v^2)^2 - (v^3)^2 $ can be positive, negative, or zero, reflecting causality.

Positions are also 4D spacetime vectors $ p $. Normalize units so $ c=1 $.

Operations: In STA, vector addition and multiplication handle boosts and rotations via rotors 
(e.g., $ R = e^{-B/2} $ for bivector $ B $), preserving the norm $ a'^2 = (Ra\tilde{R})^2 = a^2 $.


# [[cosh φ, sinh φ,     0,       0],
#  [sinh φ, cosh φ,     0,       0],
#  [0,      0,      cos φ,  −sin φ],
#  [0,      0,      sin φ,   cos φ]]