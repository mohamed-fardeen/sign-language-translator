# signlang API reference

Base URL (Render): `https://<your-service>.onrender.com`
Local: `http://localhost:8000`

## Authentication

`POST /v1/auth/token`

```json
{ "device_key": "any-8-char-or-longer-key" }
```

```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 900
}
```

The token is HS256-signed and expires in 15 minutes. Use it as
`Authorization: Bearer <token>` on protected routes.

## Health

`GET /v1/health` -- no auth.

```json
{ "status": "ok", "model_loaded": true, "model_version": "signlang@latest", "device": "cpu" }
```

## Predict

`POST /v1/predict` -- JWT required.

```json
{
  "clip": {
    "pose": [[0.1, 0.2, 0.0, ...], ...],   // shape (T, 99)
    "lh":   [[...]],                       // (T, 63)
    "rh":   [[...]],                       // (T, 63)
    "mask": [true, true, ...]              // (T,)
  },
  "top_k": 5,
  "beam_size": 1
}
```

T is padded/truncated to 64. The response:

```json
{
  "gloss_id": 17,
  "gloss_label": "thanks",
  "confidence": 0.83,
  "top_k": [
    { "id": 17, "label": "thanks", "prob": 0.83 },
    { "id": 4,  "label": "yes",    "prob": 0.09 }
  ],
  "latency_ms": 42.1,
  "model_version": "signlang@latest"
}
```

`beam_size` defaults to 1 (greedy). Set to 8 for vocabulary-constrained
beam search on the server.

## Models

`GET /v1/models` -- JWT required.

```json
[
  { "name": "signlang", "version": "latest", "backend": "torchscript", "device": "cpu" }
]
```

## Metrics

`GET /metrics` -- Prometheus text format.
