"""Serve an AdaptiveClassifier over HTTP.

    pip install "adaptive-classifier[serve]"
    python -m adaptive_classifier.serving ./my_classifier --port 8000

or, from Python::

    from adaptive_classifier.serving import create_app
    app = create_app(classifier)             # any ASGI server can run it

Endpoints: ``GET /health``, ``GET /info``, ``POST /predict``,
``POST /predict_batch``, ``POST /predict_set``, ``POST /ood``, and, only when
updates are allowed, ``POST /examples``, ``POST /calibrate``, ``POST /forget``
and ``POST /remove_examples``.

The classifier takes a lock around every operation, so one process is safe to
share across requests; scale throughput with several processes (for example
containers behind a load balancer, each with its own copy and its own updates)
rather than threads. Updates therefore only reach the process that handled
them. Run a single process if you enable them, and pass ``--save-dir`` to
persist what was learned.
"""

import argparse
import asyncio
import hmac
import logging
import math
import threading
from typing import Any, Dict, List, Optional

from .classifier import AdaptiveClassifier

logger = logging.getLogger(__name__)


def create_app(
    classifier: AdaptiveClassifier,
    *,
    allow_updates: bool = False,
    api_key: Optional[str] = None,
    max_batch: int = 1000,
    max_text_chars: int = 20_000,
    save_dir: Optional[str] = None,
):
    """Build a FastAPI app around `classifier`.

    Args:
        classifier: A trained (or loaded) classifier.
        allow_updates: Enable ``/examples``, ``/calibrate``, ``/forget`` and
            ``/remove_examples``.
        api_key: If set, update endpoints require ``Authorization: Bearer <key>``.
            Strongly recommended with ``allow_updates``.
        max_batch: Largest accepted batch (texts per request).
        max_text_chars: Longest accepted text, in characters.
        save_dir: If set, the classifier is saved there after every update. Saves
            hold every retained example, and the ONNX encoder is exported only
            the first time (it never changes).
    """
    from .multilabel import MultiLabelAdaptiveClassifier
    if isinstance(classifier, MultiLabelAdaptiveClassifier):
        raise ValueError(
            "The server does not support MultiLabelAdaptiveClassifier: its predictions and "
            "updates use lists of labels per text, which these endpoints do not express")
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field
    except ImportError as error:          # pragma: no cover - depends on the environment
        raise ImportError(
            'Serving needs FastAPI: pip install "adaptive-classifier[serve]"'
        ) from error

    if allow_updates and not api_key:
        logger.warning("Updates are enabled without an api_key: anyone who can reach this "
                       "server can change the classifier")

    class Prediction(BaseModel):
        label: str
        score: float

    class PredictRequest(BaseModel):
        text: str = Field(min_length=1, max_length=max_text_chars)
        k: int = Field(default=5, ge=0, le=1000)
        abstain_below: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        abstain_ood: bool = False

    class PredictResponse(BaseModel):
        predictions: List[Prediction]
        abstained: bool

    class BatchRequest(BaseModel):
        texts: List[str] = Field(min_length=1, max_length=max_batch)
        k: int = Field(default=5, ge=0, le=1000)
        ood: bool = False
        ood_threshold: Optional[float] = Field(default=None, gt=0.0)

    class OodResult(BaseModel):
        ood_score: Optional[float]      # None when the classifier has no classes
        ood: bool

    class BatchResponse(BaseModel):
        predictions: List[List[Prediction]]
        # Only present when the request asked for "ood": true; index-aligned with predictions.
        ood: Optional[List[OodResult]] = None
        ood_threshold: Optional[float] = None

    class SetRequest(BaseModel):
        text: str = Field(min_length=1, max_length=max_text_chars)
        alpha: float = Field(default=0.1, gt=0.0, lt=1.0)

    class OodRequest(BaseModel):
        text: str = Field(min_length=1, max_length=max_text_chars)
        threshold: Optional[float] = Field(default=None, gt=0.0)

    class ExamplesRequest(BaseModel):
        texts: List[str] = Field(min_length=1, max_length=max_batch)
        labels: List[str] = Field(min_length=1, max_length=max_batch)

    class CalibrateRequest(BaseModel):
        texts: List[str] = Field(min_length=2, max_length=max_batch)
        labels: List[str] = Field(min_length=2, max_length=max_batch)

    class ForgetRequest(BaseModel):
        labels: List[str] = Field(min_length=1, max_length=max_batch)

    class RemoveRequest(BaseModel):
        texts: List[str] = Field(min_length=1, max_length=max_batch)
        label: Optional[str] = None

    app = FastAPI(title="Adaptive Classifier", version=_version())

    @app.exception_handler(ValueError)
    async def bad_request(_: Request, error: ValueError):
        # The classifier raises ValueError for unusable input (unknown label, empty text, ...).
        return JSONResponse(status_code=400, content={"detail": str(error)})

    def pairs(items) -> List[Dict[str, Any]]:
        return [{"label": label, "score": float(score)} for label, score in items]

    def require_updates(request: Request):
        if not allow_updates:
            raise HTTPException(status_code=403, detail="Updates are disabled on this server")
        if api_key:
            header = request.headers.get("authorization", "")
            supplied = header[7:] if header.lower().startswith("bearer ") else ""
            if not hmac.compare_digest(supplied.encode(), api_key.encode()):
                raise HTTPException(status_code=401, detail="Missing or invalid API key",
                                    headers={"WWW-Authenticate": "Bearer"})

    save_lock = threading.Lock()

    def save():
        # One save at a time: each file is replaced atomically, but two interleaved saves
        # could still leave files from different saves side by side.
        with save_lock:
            classifier.save(save_dir, all_examples=True)

    async def persist():
        # The first save exports the ONNX encoder (seconds); later ones write only the
        # learned state. Either way keep it off the event loop.
        if save_dir:
            await asyncio.to_thread(save)

    def classes():
        # Read under the classifier's lock: an update running in another thread rewrites the label map.
        with classifier._lock:
            return sorted(classifier.label_to_id)

    def snapshot():
        with classifier._lock:
            config = classifier.config
            return {
                "classes": sorted(classifier.label_to_id),
                "examples_per_class": classifier.get_memory_stats().get("examples_per_class", {}),
                "calibrated": bool(classifier.calibration),
                "last_calibrated_at": classifier.last_calibrated_at,
                "examples_since_calibration": classifier.examples_since_calibration,
                "max_examples_per_class": config.max_examples_per_class,
                "retention_recent_fraction": config.retention_recent_fraction,
                "ood_threshold": config.ood_threshold,
                "ood_radius_percentile": config.ood_radius_percentile,
            }

    @app.get("/health")
    async def health():
        return {"status": "ok", "classes": len(classifier.label_to_id)}

    @app.get("/info")
    async def info():
        state = await asyncio.to_thread(snapshot)
        return {
            "version": _version(),
            "model": getattr(classifier, "_model_name", None),
            **state,
            "updates_allowed": allow_updates,
            "max_batch": max_batch,
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(body: PredictRequest):
        result = await classifier.apredict(
            body.text, body.k, abstain_below=body.abstain_below, abstain_ood=body.abstain_ood
        )
        asked_to_abstain = body.abstain_below is not None or body.abstain_ood
        abstained = asked_to_abstain and not result and body.k > 0
        return {"predictions": pairs(result), "abstained": abstained}

    @app.post("/predict_batch", response_model=BatchResponse, response_model_exclude_unset=True)
    async def predict_batch(body: BatchRequest):
        if any(not t or len(t) > max_text_chars for t in body.texts):
            raise HTTPException(status_code=422, detail=f"Each text must be 1-{max_text_chars} characters")
        if not body.ood:
            result = await classifier.apredict_batch(body.texts, body.k)
            return {"predictions": [pairs(r) for r in result]}
        # One encoder pass serves both the predictions and the OOD scores.
        result, scores = await classifier.apredict_batch_with_ood(body.texts, body.k)
        limit = classifier.config.ood_threshold if body.ood_threshold is None else body.ood_threshold
        return {
            "predictions": [pairs(r) for r in result],
            "ood": [{"ood_score": None if math.isinf(score) else float(score), "ood": bool(score > limit)}
                    for score in scores],
            "ood_threshold": limit,
        }

    @app.post("/predict_set")
    async def predict_set(body: SetRequest):
        result = await asyncio.to_thread(classifier.predict_set, body.text, body.alpha)
        return {"labels": pairs(result)}

    @app.post("/ood")
    async def ood(body: OodRequest):
        score = await asyncio.to_thread(classifier.ood_score, body.text)
        limit = classifier.config.ood_threshold if body.threshold is None else body.threshold
        return {"score": None if score == float("inf") else score, "threshold": limit,
                "ood": score > limit}

    @app.post("/examples", dependencies=[Depends(require_updates)])
    async def add_examples(body: ExamplesRequest):
        if any(not t or len(t) > max_text_chars for t in body.texts):
            raise HTTPException(status_code=422, detail=f"Each text must be 1-{max_text_chars} characters")
        await classifier.aadd_examples(body.texts, body.labels)
        await persist()
        return {"added": len(body.texts), "classes": await asyncio.to_thread(classes)}

    @app.post("/calibrate", dependencies=[Depends(require_updates)])
    async def calibrate(body: CalibrateRequest):
        if any(not t or len(t) > max_text_chars for t in body.texts):
            raise HTTPException(status_code=422, detail=f"Each text must be 1-{max_text_chars} characters")
        report = await asyncio.to_thread(classifier.calibrate, body.texts, body.labels)
        await persist()
        return report

    @app.post("/forget", dependencies=[Depends(require_updates)])
    async def forget(body: ForgetRequest):
        await asyncio.to_thread(classifier.forget, body.labels)
        await persist()
        return {"classes": await asyncio.to_thread(classes)}

    @app.post("/remove_examples", dependencies=[Depends(require_updates)])
    async def remove_examples(body: RemoveRequest):
        removed = await asyncio.to_thread(classifier.remove_examples, body.texts, body.label)
        await persist()
        return {"removed": removed, "classes": await asyncio.to_thread(classes)}

    return app


def _version() -> str:
    from . import __version__
    return __version__


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m adaptive_classifier.serving",
        description="Serve a saved AdaptiveClassifier over HTTP.",
    )
    parser.add_argument("model", help="Directory from classifier.save(), or a Hub id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-updates", action="store_true",
                        help="Enable /examples, /calibrate, /forget and /remove_examples")
    parser.add_argument("--api-key", default=None,
                        help="Bearer token required for updates (or set ADAPTIVE_CLASSIFIER_API_KEY)")
    parser.add_argument("--save-dir", default=None, help="Save the classifier here after every update")
    parser.add_argument("--max-batch", type=int, default=1000,
                        help="Most texts accepted per request (default 1000)")
    parser.add_argument("--max-examples-per-class", type=int, default=None,
                        help="Keep at most this many examples per class (newest plus boundary examples)")
    parser.add_argument("--retention-recent-fraction", type=float, default=None,
                        help="Share of --max-examples-per-class kept for the newest examples (default 0.5); "
                             "the rest keeps the examples closest to another class")
    parser.add_argument("--ood-radius-percentile", type=float, default=None,
                        help="Percentile of training distances that defines a class's OOD radius (default 95)")
    parser.add_argument("--no-onnx", action="store_true", help="Use PyTorch even on CPU")
    args = parser.parse_args(argv)

    import os
    try:
        import uvicorn
    except ImportError as error:          # pragma: no cover
        raise SystemExit('uvicorn is required: pip install "adaptive-classifier[serve]"') from error

    if args.max_batch < 1:
        parser.error("--max-batch must be at least 1")
    if args.ood_radius_percentile is not None and not 0 < args.ood_radius_percentile <= 100:
        parser.error("--ood-radius-percentile must be in (0, 100]")
    if args.max_examples_per_class is not None and args.max_examples_per_class < 1:
        parser.error("--max-examples-per-class must be at least 1")
    if args.retention_recent_fraction is not None and not 0 <= args.retention_recent_fraction <= 1:
        parser.error("--retention-recent-fraction must be in [0, 1]")

    classifier = AdaptiveClassifier.load(args.model, use_onnx=False if args.no_onnx else "auto")
    if args.ood_radius_percentile is not None:
        classifier.config.ood_radius_percentile = args.ood_radius_percentile
    if args.retention_recent_fraction is not None:
        classifier.config.retention_recent_fraction = args.retention_recent_fraction
    if args.max_examples_per_class is not None:
        evicted = classifier.set_max_examples_per_class(args.max_examples_per_class)
        if evicted:
            logger.info(f"Evicted {evicted} examples to fit --max-examples-per-class")
    app = create_app(
        classifier,
        allow_updates=args.allow_updates,
        api_key=args.api_key or os.environ.get("ADAPTIVE_CLASSIFIER_API_KEY"),
        max_batch=args.max_batch,
        save_dir=args.save_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
