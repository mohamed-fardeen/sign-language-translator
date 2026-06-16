from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from signlang.config import load_config, to_dict
from signlang.data.datamodules import SignDataModule
from signlang.evaluation.metrics import topk_accuracy
from signlang.evaluation.reports import classification_report, write_report
from signlang.inference.postprocess import greedy_decode
from signlang.models.sign_model import SignModel
from signlang.utils.io import read_json
from signlang.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split")
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--out", default=None, help="Path to write test_metrics.json")
    args = parser.parse_args()

    configure_logging("INFO", json_logs=False)
    log = get_logger("evaluate")

    cfg = to_dict(load_config())
    vocab = read_json(Path(cfg["paths"]["vocab_path"]))
    vocab.get("gloss_to_id", {})
    id_to_gloss = vocab.get("id_to_gloss", {})
    label_names = {int(k): v for k, v in id_to_gloss.items()}

    dm = SignDataModule(features_dir=args.features_dir, batch_size=32, num_workers=2, augment_train=False)
    dm.setup("test")

    model = SignModel.load_from_checkpoint(args.ckpt, cfg=cfg["model"], map_location="cpu")
    model.eval()

    preds: list[int] = []
    targets: list[int] = []
    with torch.inference_mode():
        for batch in DataLoader(dm.test_ds, batch_size=32, num_workers=2):
            out = model(batch["pose"], batch["lh"], batch["rh"], batch["face"], mask=batch.get("mask"))
            decoded = greedy_decode(out.logits, blank=0)
            for ids in decoded:
                preds.append(ids[0] if ids else 0)
            for t in batch["label"].tolist():
                targets.append(t)

    report = classification_report(preds, targets, label_names=label_names)
    report["top1"] = topk_accuracy(preds, targets, k=1)
    report["top5_approx"] = topk_accuracy(preds, targets, k=5)
    out_path = Path(args.out) if args.out else Path(cfg["paths"]["log_dir"]) / "test_metrics.json"
    write_report(report, out_path)
    log.info("evaluate.done", accuracy=report["accuracy"], out=str(out_path))


if __name__ == "__main__":
    main()
