# Checkpointing

Training checkpoints are stored in a safetensors-first format. Each checkpoint
is a directory containing a `manifest.json` file and a set of tensor and metadata
files. Use the `resume` flag to control how training starts:

- `resume=auto` (default): resume from the latest checkpoint if available, otherwise
  start from scratch.
- `resume=none`: ignore existing checkpoints and start fresh.
- `resume=PATH`: load the specific checkpoint manifest located at `PATH`.

Determinism vs. speed when resuming:

- `deterministic_resume=true` (default): the loader consumes the exact number of
  previously completed batches to reproduce the same data order before continuing.
  This guarantees bitwise equivalence with an uninterrupted run but can be slow for
  very large `step` counts.
- `deterministic_resume=false`: resumes immediately from the next available batch
  without consuming past batches. Faster, but the data order may differ slightly.

Checkpoints are written atomically and `latest.json` in the checkpoint root points
to the most recent complete checkpoint. An `lkg.json` file keeps a pointer to the
last known good checkpoint.

All tensor data (model weights, optimizer state and RNG state) are stored using
[safetensors](https://github.com/huggingface/safetensors) for safety and portability.
Metadata such as optimizer parameters and run context are stored in JSON.
