import os
import json
import time
import hashlib
import tempfile
from typing import Optional, Dict, Any, Sequence

import random
import numpy as np

import torch
from safetensors.torch import save_file, load_file


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CheckpointIO:
    """Safetensors based checkpoint writer/reader."""

    format_version = 1

    def __init__(self, root_dir: str, run_subdir: str = "sft-v1", sync_dir_fsync: bool = True) -> None:
        self.root_dir = root_dir
        self.run_dir = os.path.join(root_dir, run_subdir)
        self.sync_dir_fsync = sync_dir_fsync
        os.makedirs(self.run_dir, exist_ok=True)
        self.latest_pointer = os.path.join(root_dir, "latest.json")
        self.lkg_pointer = os.path.join(root_dir, "lkg.json")

    # ------------------------------------------------------------------
    def _write_atomic(self, path: str, data: bytes) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            if self.sync_dir_fsync:
                _fsync_dir(os.path.dirname(path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _hash_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizers: Sequence[torch.optim.Optimizer],
        step: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write a complete checkpoint and return manifest path."""

        timestamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
        step_dir = os.path.join(self.run_dir, f"{timestamp}_step{step:06d}")
        tmp_dir = step_dir + ".tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        # Model
        model_path = os.path.join(tmp_dir, "model.safetensors")
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save_file(cpu_state, model_path)

        # Optimizers
        opt_tensor: Dict[str, torch.Tensor] = {}
        opt_meta: Dict[str, Any] = {"optimizers": []}
        for idx, opt in enumerate(optimizers):
            state = opt.state_dict()
            meta_state: Dict[str, Any] = {"state": {}, "param_groups": state["param_groups"]}
            for p_id, s in state["state"].items():
                for k, v in s.items():
                    key = f"{idx}.{p_id}.{k}"
                    if torch.is_tensor(v):
                        opt_tensor[key] = v.detach().cpu()
                    else:
                        meta_state["state"].setdefault(str(p_id), {})[k] = v
            opt_meta["optimizers"].append(meta_state)
        opt_tensor_path = os.path.join(tmp_dir, "optimizer.safetensors")
        if len(opt_tensor):
            save_file(opt_tensor, opt_tensor_path)
        with open(os.path.join(tmp_dir, "optimizer.json"), "w", encoding="utf-8") as f:
            json.dump(opt_meta, f)

        # RNG state
        rng_tensor: Dict[str, torch.Tensor] = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            try:
                for i, t in enumerate(torch.cuda.get_rng_state_all()):
                    rng_tensor[f"cuda_{i}"] = t
            except RuntimeError:
                pass
        py_state = random.getstate()
        np_state = np.random.get_state()
        rng_meta = {
            "python": [py_state[0], list(py_state[1]), py_state[2]],
            "numpy": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
        }
        rng_tensor_path = os.path.join(tmp_dir, "rng.safetensors")
        save_file(rng_tensor, rng_tensor_path)
        with open(os.path.join(tmp_dir, "rng.json"), "w", encoding="utf-8") as f:
            json.dump(rng_meta, f)

        # Extra metadata
        manifest: Dict[str, Any] = {
            "format_version": self.format_version,
            "created_utc": timestamp,
            "step": step,
            "components": {
                "model": {"path": "model.safetensors"},
                "optimizer": {"path": "optimizer.safetensors" if len(opt_tensor) else None},
                "rng": {"path": "rng.safetensors"},
            },
            "extra": extra or {},
        }

        # Checksums
        checksums = {}
        for name, comp in manifest["components"].items():
            path = comp.get("path")
            if path:
                full = os.path.join(tmp_dir, path)
                checksums[name] = {"sha256": self._hash_file(full), "size": os.path.getsize(full)}
        manifest["checksums"] = checksums

        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Fsync everything
        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                p = os.path.join(root, fn)
                with open(p, "rb") as f:
                    os.fsync(f.fileno())
        if self.sync_dir_fsync:
            _fsync_dir(tmp_dir)

        os.replace(tmp_dir, step_dir)
        if self.sync_dir_fsync:
            _fsync_dir(self.run_dir)

        # Update pointers
        if os.path.exists(self.latest_pointer):
            data = open(self.latest_pointer, "rb").read()
            self._write_atomic(self.lkg_pointer, data)
        self._write_atomic(self.latest_pointer, json.dumps({"manifest": os.path.join(step_dir, "manifest.json")}).encode("utf-8"))

        return os.path.join(step_dir, "manifest.json")

    # ------------------------------------------------------------------
    def resolve_latest_or_none(self) -> Optional[str]:
        try:
            with open(self.latest_pointer, "r", encoding="utf-8") as f:
                data = json.load(f)
            manifest = data.get("manifest")
            if manifest and os.path.exists(manifest):
                return manifest
        except Exception:
            return None
        return None

    def any_checkpoint_exists(self) -> bool:
        return self.resolve_latest_or_none() is not None

    def resolve_path_or_none(self, user_supplied: str) -> Optional[str]:
        if user_supplied is None:
            return None
        if os.path.isdir(user_supplied):
            manifest = os.path.join(user_supplied, "manifest.json")
        else:
            manifest = user_supplied
        return manifest if os.path.exists(manifest) else None

    # ------------------------------------------------------------------
    def load(
        self,
        target: Optional[str],
        model: torch.nn.Module,
        map_location: torch.device,
        strict: bool = True,
        allow_degraded: bool = True,
        optimizers: Optional[Sequence[torch.optim.Optimizer]] = None,
    ) -> Dict[str, Any]:
        if target is None:
            return {"step": 0, "missing": ["manifest"], "warnings": ["no checkpoint"]}
        with open(target, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        base_dir = os.path.dirname(target)
        missing: list[str] = []
        # Model
        model_path = os.path.join(base_dir, manifest["components"]["model"]["path"])
        tensors = load_file(model_path, device=map_location)
        model.load_state_dict(tensors, strict=strict)

        # Optimizers
        loaded_opts = []
        if optimizers is not None:
            opt_comp = manifest["components"].get("optimizer")
            if opt_comp and opt_comp.get("path"):
                opt_tensor = load_file(os.path.join(base_dir, opt_comp["path"]), device="cpu")
                with open(os.path.join(base_dir, "optimizer.json"), "r", encoding="utf-8") as f:
                    opt_meta = json.load(f)
                offset = 0
                for idx, opt in enumerate(optimizers):
                    meta_state = opt_meta["optimizers"][idx]
                    state: Dict[int, Dict[str, Any]] = {}
                    for key, tensor in opt_tensor.items():
                        oid, p_id, k = key.split(".")
                        if int(oid) != idx:
                            continue
                        state.setdefault(int(p_id), {})[k] = tensor
                    for p_id, s in meta_state["state"].items():
                        state[int(p_id)].update(s)
                    full_state = {"state": state, "param_groups": meta_state["param_groups"]}
                    opt.load_state_dict(full_state)
                    loaded_opts.append(full_state)
            else:
                missing.append("optimizer")
        else:
            missing.append("optimizer")

        # RNG
        rng_comp = manifest["components"].get("rng")
        if rng_comp and rng_comp.get("path"):
            rng_tensors = load_file(os.path.join(base_dir, rng_comp["path"]), device="cpu")
            torch.set_rng_state(rng_tensors["cpu"])
            cuda_states = [rng_tensors[k] for k in sorted(rng_tensors.keys()) if k.startswith("cuda_")]
            if cuda_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
            with open(os.path.join(base_dir, "rng.json"), "r", encoding="utf-8") as f:
                rng_meta = json.load(f)
            py_state = rng_meta.get("python")
            if py_state is not None:
                random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))
            np_state = rng_meta.get("numpy")
            if np_state is not None:
                np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), np_state[2], np_state[3], np_state[4]))
        else:
            missing.append("rng")

        return {"step": manifest.get("step", 0), "extra": manifest.get("extra", {}), "missing": missing, "warnings": []}
