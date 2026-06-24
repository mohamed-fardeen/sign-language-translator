from __future__ import annotations

import time

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from signlang.serving.model_registry import pad_or_truncate
from signlang.serving.schemas.request import (
    PredictionItem,
    PredictRequest,
    PredictResponse,
)
from signlang.serving.security import require_auth
from signlang.utils.logging import get_logger

router = APIRouter(prefix="/v1", tags=["predict"])
log = get_logger(__name__)


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request, _claims=Depends(require_auth)) -> PredictResponse:
    cfg = request.app.state.predict_cfg
    predictor = request.app.state.registry.active
    vocab = request.app.state.vocab
    id_to_gloss = {int(k): v for k, v in vocab.get("id_to_gloss", {}).items()}

    pose = np.asarray(req.clip.pose, dtype=np.float32)
    lh = np.asarray(req.clip.lh, dtype=np.float32)
    rh = np.asarray(req.clip.rh, dtype=np.float32)
    mask = np.asarray(req.clip.mask, dtype=bool) if req.clip.mask is not None else None

    t = int(cfg.get("clip_frames", 64))
    pose = pad_or_truncate(pose, t)
    lh = pad_or_truncate(lh, t)
    rh = pad_or_truncate(rh, t)
    mask = np.ones(t, dtype=bool) if mask is None else pad_or_truncate(mask, t)

    if pose.shape != (t, 99) or lh.shape != (t, 63) or rh.shape != (t, 63):
        raise HTTPException(
            status_code=422,
            detail=f"Clip shape mismatch: pose={pose.shape} lh={lh.shape} rh={rh.shape}",
        )

    start = time.perf_counter()
    device = next(predictor.model.parameters()).device
    with torch.inference_mode():
        p = torch.from_numpy(pose).unsqueeze(0).to(device).float()
        l = torch.from_numpy(lh).unsqueeze(0).to(device).float()
        r = torch.from_numpy(rh).unsqueeze(0).to(device).float()
        logits = predictor.model(p, l, r)  # (1, num_classes)

        # CTC kept for reference (removed in v1 single-label mode):
        # beam = req.beam_size or int(cfg.get("beam_size", 1))
        # if beam > 1:
        #     log_probs = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()
        #     ids = beam_search_decode(log_probs, beam_size=beam, blank=0)
        # else:
        #     ids = greedy_decode(logits, blank=0)[0]
        # probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    latency_ms = (time.perf_counter() - start) * 1000.0

    # Classification: argmax over num_classes, add 1 to recover the
    # manifest label (manifest labels are 1-based).
    pred_idx = int(probs.argmax())
    manifest_label = pred_idx + 1

    order = probs.argsort()[::-1][: req.top_k]
    top_k = [
        PredictionItem(
            id=int(i) + 1,
            label=id_to_gloss.get(int(i) + 1, str(int(i) + 1)),
            prob=float(probs[i]),
        )
        for i in order
    ]

    return PredictResponse(
        gloss_id=manifest_label,
        gloss_label=id_to_gloss.get(manifest_label, str(manifest_label)),
        confidence=float(probs[pred_idx]) if top_k else 0.0,
        top_k=top_k,
        latency_ms=round(latency_ms, 3),
        model_version=request.app.state.registry.active_key,
    )