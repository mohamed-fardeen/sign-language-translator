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
    parser = argparse.ArgumentParser(description="Export model to ONNX (FP16) and INT8")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--no-int8", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    log = get_logger("export_onnx")

    cfg = to_dict(load_config())
    opset = args.opset or int(cfg["model_export"]["onnx_opset"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SignModel.load_from_checkpoint(args.ckpt, cfg=cfg["model"], map_location="cpu")
    model.eval()
    wrapped = _Wrapper(model).eval()

    clip_frames = int(cfg["model"]["clip_frames"])
    pose = torch.randn(1, clip_frames, int(cfg["model"]["in_dim_pose"]))
    lh = torch.randn(1, clip_frames, int(cfg["model"]["in_dim_hand"]))
    rh = torch.randn(1, clip_frames, int(cfg["model"]["in_dim_hand"]))
    face = torch.randn(1, clip_frames, int(cfg["model"]["in_dim_face"]))

    fp16_path = out_dir / "model_fp16.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            wrapped,
            (pose, lh, rh, face),
            str(fp16_path),
            opset_version=opset,
            input_names=["pose", "lh", "rh", "face"],
            output_names=["logits"],
            dynamic_axes={
                "pose": {0: "batch", 1: "frames"},
                "lh": {0: "batch", 1: "frames"},
                "rh": {0: "batch", 1: "frames"},
                "face": {0: "batch", 1: "frames"},
                "logits": {0: "batch", 1: "frames"},
            },
        )
    log.info("onnx.fp16.exported", path=str(fp16_path))

    if not args.no_int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            int8_path = out_dir / "model_int8.onnx"
            quantize_dynamic(
                model_input=str(fp16_path),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gemm", "Attention"],
            )
            log.info("onnx.int8.exported", path=str(int8_path))
        except Exception as exc:
            log.error("onnx.int8.failed", error=str(exc))


if __name__ == "__main__":
    main()
