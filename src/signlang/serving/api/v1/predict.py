from __future__ import annotations

import time

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from signlang.inference.postprocess import beam_search_decode, greedy_decode
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
    face = np.asarray(req.clip.face, dtype=np.float32)
    mask = np.asarray(req.clip.mask, dtype=bool) if req.clip.mask is not None else None

    t = int(cfg.get("clip_frames", 64))
    pose = pad_or_truncate(pose, t)
    lh = pad_or_truncate(lh, t)
    rh = pad_or_truncate(rh, t)
    face = pad_or_truncate(face, t)
    mask = np.ones(t, dtype=bool) if mask is None else pad_or_truncate(mask, t)

    if pose.shape != (t, 99) or lh.shape != (t, 63) or rh.shape != (t, 63) or face.shape != (t, 120):
        raise HTTPException(
            status_code=422,
            detail=f"Clip shape mismatch: pose={pose.shape} lh={lh.shape} rh={rh.shape} face={face.shape}",
        )

    start = time.perf_counter()
    device = next(predictor.model.parameters()).device
    with torch.inference_mode():
        p = torch.from_numpy(pose).unsqueeze(0).to(device).float()
        l = torch.from_numpy(lh).unsqueeze(0).to(device).float()
        r = torch.from_numpy(rh).unsqueeze(0).to(device).float()
        f = torch.from_numpy(face).unsqueeze(0).to(device).float()
        out = predictor.model(p, l, r, f)
        if isinstance(out, tuple):
            logits = out[0]
        elif isinstance(out, dict):
            logits = out["logits"]
        else:
            logits = out
        beam = req.beam_size or int(cfg.get("beam_size", 1))
        if beam > 1:
            log_probs = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()
            ids = beam_search_decode(log_probs, beam_size=beam, blank=0)
        else:
            ids = greedy_decode(logits, blank=0)[0]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    latency_ms = (time.perf_counter() - start) * 1000.0

    top = probs.mean(axis=0)
    order = top.argsort()[::-1][: req.top_k]
    top_k = [
        PredictionItem(
            id=int(i),
            label=id_to_gloss.get(int(i), str(int(i))),
            prob=float(top[i]),
        )
        for i in order
    ]
    gloss_id = ids[0] if ids else 0
    return PredictResponse(
        gloss_id=int(gloss_id),
        gloss_label=id_to_gloss.get(int(gloss_id), str(int(gloss_id))),
        confidence=float(top_k[0].prob) if top_k else 0.0,
        top_k=top_k,
        latency_ms=round(latency_ms, 3),
        model_version=request.app.state.registry.active_key,
    )
