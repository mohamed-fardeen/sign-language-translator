from __future__ import annotations

import argparse
from pathlib import Path

import torch

from signlang.config import load_config, to_dict
from signlang.models.sign_model import SignModel
from signlang.utils.logging import configure_logging, get_logger


class _Wrapper(torch.nn.Module):
    def __init__(self, model: SignModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, pose, lh, rh, face):
        out = self.model(pose, lh, rh, face)
        return out.logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to TorchScript (FP16)")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    log = get_logger("export_torchscript")

    cfg = to_dict(load_config())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SignModel.load_from_checkpoint(args.ckpt, cfg=cfg["model"], map_location="cpu")
    model.eval()
    wrapped = _Wrapper(model).eval()

    if not args.no_fp16:
        wrapped = wrapped.half()

    clip_frames = int(cfg["model"]["clip_frames"])
    pose = torch.randn(1, clip_frames, int(cfg["model"]["in_dim_pose"]))
    lh = torch.randn(1, clip_frames, int(cfg["model"]["in_dim_hand"]))
    rh = torch.randn(1, clip_frames, int(cfg["model"]["in_dim_hand"]))

    traced = torch.jit.trace(wrapped, (pose, lh, rh), strict=False)
    out_path = out_dir / "model.pt"
    torch.jit.save(traced, str(out_path))
    log.info("torchscript.exported", path=str(out_path))


if __name__ == "__main__":
    main()
