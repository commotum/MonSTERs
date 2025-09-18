from typing import List, Optional
import yaml
import os
import re

import torch
import torch.distributed as dist
from safetensors.torch import load_file as safetensors_load_file  # type: ignore

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]


def _find_run_root(start_path: str) -> Optional[str]:
    """Ascend from start_path until a directory containing all_config.yaml is found."""
    cur = os.path.abspath(start_path)
    # If start_path is a file, go to its directory
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    while True:
        if os.path.exists(os.path.join(cur, "all_config.yaml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _resolve_checkpoint_components(path: str):
    """Return (weights_path, run_root, step_int) supporting both .pt and safetensors layouts.

    - If `path` is a directory that contains manifest.json and model.safetensors, use that model.safetensors.
    - If `path` is a safetensors file, use it and search upwards for all_config.yaml.
    - If `path` is a torch checkpoint file, use it and search upwards for all_config.yaml.
    - Step number is parsed from basename if it starts with step_XXXXXX, otherwise from parent directory name suffix _stepXXXXXX.
    """
    ap = os.path.abspath(path)
    weights_path = ap
    step = 0

    # If directory, try safetensors layout
    if os.path.isdir(ap):
        maybe_model = os.path.join(ap, "model.safetensors")
        if os.path.exists(maybe_model):
            weights_path = maybe_model
        # step from dir name like ..._step287219
        m = re.search(r"_step(\d+)$", os.path.basename(ap))
        if m:
            step = int(m.group(1))
    else:
        # If file, infer step from file name or its parent dir
        base = os.path.basename(ap)
        if base.startswith("step_"):
            try:
                step = int(base.split("_", 1)[1])
            except Exception:
                step = 0
        else:
            m = re.search(r"_step(\d+)$", os.path.basename(os.path.dirname(ap)))
            if m:
                step = int(m.group(1))

    run_root = _find_run_root(weights_path)
    if run_root is None:
        # Fallback: try one level up from step dir
        run_root = _find_run_root(os.path.dirname(os.path.dirname(weights_path)))
    return weights_path, run_root, step


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore

    RANK = 0
    WORLD_SIZE = 1
    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        # Validate device availability before setting
        local_rank = int(os.environ["LOCAL_RANK"])
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK {local_rank} is >= available CUDA devices ({device_count}). "
                "Set --nproc-per-node to your GPU count or adjust CUDA_VISIBLE_DEVICES."
            )
        torch.cuda.set_device(local_rank)

    weights_path, run_root, step_from_path = _resolve_checkpoint_components(eval_cfg.checkpoint)
    if run_root is None:
        raise FileNotFoundError(
            f"Could not locate all_config.yaml from checkpoint path: {eval_cfg.checkpoint}. "
            "Ensure you pass either the raw torch checkpoint file, the safetensors model file, or the step directory."
        )

    with open(os.path.join(run_root, "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
        config.eval_save_outputs = eval_cfg.save_outputs
        # Save predictions alongside the step directory (if provided), else in the checkpoint directory
        # If weights_path is the safetensors file, use its directory for outputs
        config.checkpoint_path = os.path.dirname(weights_path)

    # Dataloader
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Load weights: support safetensors and torch checkpoints
    try:
        if weights_path.endswith(".safetensors"):
            state = safetensors_load_file(weights_path, device="cuda")
        else:
            state = torch.load(weights_path, map_location="cuda")
        try:
            train_state.model.load_state_dict(state, assign=True)
        except Exception:
            # Handle torch.compile prefix
            train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()}, assign=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {weights_path}: {e}")

    train_state.step = step_from_path or 0

    # Evaluate
    print("Starting evaluation")

    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if metrics is not None:
        print(metrics)

    # Clean up distributed resources if initialized
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    launch()
