# MonSTERs And Friends

-----

## **Refactoring Plan: Migrating to a Coordinate-Aware Architecture**

This document outlines a step-by-step plan to refactor the repository from its current "flatten-then-apply" architecture to a modular, flexible system that uses explicit multi-dimensional coordinates.

### **Phase 1: The Data Foundation**

-----

#### **Step 1: Update Dataset Builders and Metadata**

  * **Key Files & Modules Involved:**
      * `dataset/common.py`
      * `dataset/build_sudoku_dataset.py` (and other dataset builders like `build_arc_dataset.py`)
  * **Conceptual Changes Required:**
    1.  **Extend Metadata:** In `dataset/common.py`, modify the `PuzzleDatasetMetadata` class to include information about the coordinate system, such as `coord_dims: int`.
    2.  **Generate Coordinate Arrays:** In each dataset builder (e.g., `build_sudoku_dataset.py`), modify the data generation logic. Alongside `inputs` and `labels`, create a `positions` NumPy array. For an `(N, S)` token array, this new array will have the shape `(N, S, coord_dims)`.
    3.  **Save Positions:** Save this new array to a `positions.npy` file in the output directory.
  * **Dependencies & Rationale:** This is the foundational step. It creates the explicit coordinate data on disk that the entire downstream pipeline will depend on.

-----

#### **Step 2: Update `PuzzleDataset` to Load Coordinates**

  * **Key Files & Modules Involved:** `puzzle_dataset.py`
  * **Conceptual Changes Required:**
    1.  **Lazy Load Positions:** In the `_lazy_load_dataset` method, add `"positions"` to the `field_mmap_modes` dictionary to ensure it's loaded via memory-mapping.
    2.  **Include in Batches:** In the `_iter_train` and `_iter_test` methods, when you slice the `inputs` and `labels` for a batch, also slice the corresponding `positions` data.
    3.  **Collate and Pad:** In `_collate_batch`, handle the new `positions` tensor, ensuring it is correctly padded and converted to a PyTorch tensor along with the rest of the batch data.
  * **Dependencies & Rationale:** This step makes the explicit coordinate data available to the `DataLoader`. The model cannot use the new data until the loader provides it in each batch.

-----

### **Phase 2: The Modular Core**

-----

#### **Step 3: Create the Unified Positional Encoding Interface**

  * **Key Files & Modules Involved:** Create a new file, e.g., `models/positional_encoders.py`.
  * **Conceptual Changes Required:**
    1.  Define a new abstract base class for all positional encoders. This class will establish a standard interface.
        ```python
        import torch
        from torch import nn

        class BasePositionalEncoder(nn.Module):
            """Abstract base class for all positional encoding modules."""
            def __init__(self, config, metadata):
                super().__init__()
                # Logic to pre-compute and cache tables as buffers

            def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                """Applies positional encoding to query and key tensors."""
                raise NotImplementedError
        ```
  * **Dependencies & Rationale:** This creates a clean, modular "plugin" architecture. It decouples the main model from the specific implementation details of any single encoding scheme.

-----

#### **Step 4: Refactor RoPE and MonSTER Modules**

  * **Key Files & Modules Involved:** `models/layers.py`, `models/positional_encoders.py`
  * **Conceptual Changes Required:**
    1.  Move the existing `RotaryEmbedding` and `MonsterEmbedding` logic from `models/layers.py` into new classes within `models/positional_encoders.py`.
    2.  Make these new classes inherit from `BasePositionalEncoder`.
    3.  Rewrite their `forward` methods to match the new interface: `forward(self, q, k, positions)`.
    4.  **Crucially, remove all logic that calculates coordinates from a 1D index** (e.g., `idx // grid_w`). The methods will now use the explicit `positions` tensor passed as an argument. The core mathematical operations for applying rotations will remain.
  * **Dependencies & Rationale:** This step adapts the existing encoders to the new, flexible standard, making them ready for modular integration.

-----

#### **Step 5: Update the `Attention` Module**

  * **Key Files & Modules Involved:** `models/layers.py`
  * **Conceptual Changes Required:**
    1.  Modify the `Attention` module's `__init__` to accept an instantiated positional encoder object (e.g., `self.pos_encoder = RoPE(...)`).
    2.  Change the `Attention.forward` signature. It will no longer accept a `cos_sin` object. Instead, it will receive the `positions` tensor from the batch.
    3.  Replace the `if/elif` block that checks the type of `cos_sin` with a single, clean call to the encoder interface: `query, key = self.pos_encoder(query, key, positions)`.
  * **Dependencies & Rationale:** This makes the `Attention` layer agnostic to the encoding strategy. It no longer needs to know about RoPE or MonSTER, only about the standard interface, simplifying the code significantly.

-----

### **Phase 3: Integration and Validation**

-----

#### **Step 6: Integrate the New System into the Main Model**

  * **Key Files & Modules Involved:** `models/hrm/hrm_act_v1.py`
  * **Conceptual Changes Required:**
    1.  In the `HierarchicalReasoningModel_ACTV1_Inner` `__init__` method, replace the `if/elif` block that creates specific embedding classes with a factory or registry that instantiates the correct `BasePositionalEncoder` subclass based on the model configuration.
    2.  Modify the model's `forward` method to accept the full `batch` dictionary, which now includes the `positions` tensor.
    3.  Pass the `positions` tensor down through the model's layers to the `Attention` modules.
  * **Dependencies & Rationale:** This step wires the new modular system into the main model architecture, completing the core refactor.

-----

#### **Step 7: Update Training and Evaluation Scripts**

  * **Key Files & Modules Involved:** `pretrain.py`, `config/**/*.yaml`
  * **Conceptual Changes Required:**
    1.  Update any configuration files with new parameters required by the encoders (e.g., `coord_dims`).
    2.  In `pretrain.py`, ensure the model is instantiated with the new metadata from the dataset.
    3.  Confirm that the training loop correctly handles the `positions` key within the batch dictionary.
  * **Dependencies & Rationale:** Ensures the end-to-end pipeline, from configuration to model training, is aware of and correctly uses the new coordinate-aware data structures.

-----

#### **Step 8: Performance and Sanity Checks**

  * **Key Files & Modules Involved:** All modified files.
  * **Conceptual Changes Required:** This is a validation step, not a code change step.
    1.  **Profile I/O:** Confirm that the `DataLoader` throughput has not been negatively impacted.
    2.  **Verify Caching:** Check that the new encoder modules still correctly pre-compute their lookup tables and register them as `nn.Buffer`s, avoiding re-computation on every forward pass.
    3.  **Run Unit Tests:** Create and run tests for the new `BasePositionalEncoder` subclasses to ensure their mathematical correctness.
  * **Dependencies & Rationale:** Guarantees that the refactor not only works correctly but has also preserved the high-performance characteristics of the original system.