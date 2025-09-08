# Checkpointing

Training checkpoints are stored in a safetensors-first format. Each checkpoint
is a directory containing a `manifest.json` file and a set of tensor and metadata
files. Use the `resume` flag to control how training starts:

- `resume=auto` (default): resume from the latest checkpoint if available, otherwise
  start from scratch.
- `resume=none`: ignore existing checkpoints and start fresh.
- `resume=PATH`: load the specific checkpoint manifest located at `PATH`.

Checkpoints are written atomically and `latest.json` in the checkpoint root points
to the most recent complete checkpoint. An `lkg.json` file keeps a pointer to the
last known good checkpoint.

All tensor data (model weights, optimizer state and RNG state) are stored using
[safetensors](https://github.com/huggingface/safetensors) for safety and portability.
Metadata such as optimizer parameters and run context are stored in JSON.
