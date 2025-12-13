Here’s a **clean, paper-ready version** with three parts: a short abstract, a longer description, and a compact table of the geometric symmetries being tested.

---

## **Short Abstract (Very Brief)**

We evaluate whether attention architectures implicitly respect core geometric symmetries by training a GPT-2–style Transformer on centered, upright MNIST digits and freezing the model. We then test zero-shot recognition under unseen transformations—translations, rotations, mirror operations (via 3D π-rotations), mild 3D tilts, and cross-resolution resampling—varying only the positional encoding (1D RoPE, 2D Axial RoPE, or MonSTERs). Performance under these tests isolates which symmetries are preserved by the **attention geometry itself**, rather than learned from data.

---

## **Longer Description**

In this experiment, all models share an identical GPT-2–style backbone and are trained exclusively on centered, upright MNIST digits, without any spatial augmentation. The only difference between models is the positional/structural encoding used within attention: classic 1D RoPE (flattened index), 2D Axial RoPE (separable x/y phases), or MonSTERs (4D spacetime displacements with Minkowski rotors).

After training, models are frozen and evaluated on a suite of *out-of-distribution geometric transformations* applied to the test set, without further fine-tuning. These transformations are chosen to probe which geometric symmetries are natively respected by the attention mechanism. Because the dataset never teaches these symmetries explicitly, success indicates that the symmetry is encoded *architecturally* rather than learned.

This setup cleanly separates data-driven invariance from inductive bias. In particular, it tests translation equivariance, rotation equivariance, orientation sensitivity (chirality), robustness to mild 3D viewpoint changes, and cross-resolution consistency—properties that are difficult or impossible to represent faithfully with 1D or axial positional encodings, but are natural in MonSTERs’ 4D spacetime geometry.

---

## **Geometric Symmetries Tested**

| Test Condition       | Transformation                              | Symmetry Being Probed               | What “Success” Means                             |
| -------------------- | ------------------------------------------- | ----------------------------------- | ------------------------------------------------ |
| **Translation**      | Integer shifts in x/y                       | Translation equivariance            | Same digit recognized regardless of location     |
| **Rotation (2D)**    | Rotations by θ ∈ {±15°, ±30°, ±45°, ±90°}   | Rotation equivariance               | Stable recognition across orientations           |
| **Mirror (via 3D)**  | π-rotation about x or y axis (z=0 plane)    | Orientation sensitivity (chirality) | Distinguish mirrored forms when semantics differ |
| **3D Tilt**          | Small rotations about x/y with fixed camera | 3D viewpoint robustness             | Graceful degradation under out-of-plane motion   |
| **Cross-Resolution** | 32×32 → 48×48 → 64×64                       | Scale / resolution consistency      | Same identity across sampling densities          |

---

**Interpretation:**
Higher zero-shot accuracy under these transformations indicates that the corresponding symmetry is preserved by the attention geometry itself. This experiment therefore measures *which symmetries are “free”*—i.e., built into the representation—rather than learned via augmentation.
