import os
from datetime import datetime

import torch


def _resolve_multimodal_projector(model):
    if hasattr(model, "multi_modal_projector"):
        return model.multi_modal_projector, "multi_modal_projector"

    if hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
        return model.model.multi_modal_projector, "model.multi_modal_projector"

    return None, None


def _make_run_dir_with_timestamp(save_root_dir, timestamp_fmt="%Y_%m_%d_%H_%M_%S"):
    timestamp = datetime.now().strftime(timestamp_fmt)
    run_dir = os.path.join(save_root_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def register_projector_capture_hook(model, enabled, save_root_dir, max_saves, image_path, prompt_text):
    if not enabled:
        return None, None

    projector, projector_name = _resolve_multimodal_projector(model)
    if projector is None:
        raise RuntimeError("Could not find `multi_modal_projector` in the loaded model.")

    run_dir = _make_run_dir_with_timestamp(save_root_dir)
    state = {"saved": 0}

    def _hook(module, inputs, output):
        if state["saved"] >= max_saves:
            return

        if not torch.is_tensor(output):
            return

        tensor = output.detach().float().cpu()
        save_ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        save_path = os.path.join(run_dir, f"projector_out_{state['saved']:04d}_{save_ts}.pt")

        payload = {
            "projector_name": projector_name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "saved_at": save_ts,
            "capture_index": state["saved"],
            "image_path": image_path,
            "prompt": prompt_text,
            "projector_output": tensor,
        }
        torch.save(payload, save_path)
        state["saved"] += 1
        print(f"[projector-hook] Saved capture {state['saved']}/{max_saves} -> {save_path}")

    handle = projector.register_forward_hook(_hook)
    print(f"[projector-hook] Hook attached to `{projector_name}`")
    print(f"[projector-hook] Run directory: {run_dir}")
    return handle, run_dir
