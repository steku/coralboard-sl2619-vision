"""FastAPI Vision Proxy Server for Synaptics Coralboard SL2619."""

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("coral_vision.server")

from config import COCO_CLASSES, HOST, IOU_THRESHOLD, LOADED_LABELS_PATH, MODEL_PATH, PORT, SCORE_THRESHOLD
from engine import TorqVisionEngine
from yolo import postprocess_yolov9, preprocess_image

npu_engine = TorqVisionEngine(model_path=MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize Torq NPU session and report loaded assets on startup."""
    if LOADED_LABELS_PATH:
        logger.info(f"Loaded labels file: {LOADED_LABELS_PATH} ({len(COCO_CLASSES)} classes)")
    else:
        logger.info(f"No custom labels file loaded. Using default COCO classes ({len(COCO_CLASSES)} classes).")

    logger.info(f"Initializing Synaptics Torq NPU session for model: {MODEL_PATH}...")
    try:
        npu_engine.load()
    except Exception as e:
        logger.critical(
            f"Fatal error loading configured model '{npu_engine.model_path}': {e}. Exiting server."
        )
        sys.exit(1)
    yield
    logger.info("Shutting down Vision Proxy Service.")


app = FastAPI(
    title="Coralboard SL2619 Vision Proxy",
    description="HTTP detector proxy bridging Frigate NVR with Synaptics Torq NPU",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check reporting NPU runner status and diagnostics."""
    return {
        "status": "healthy" if npu_engine.is_ready else "error",
        "npu_ready": npu_engine.is_ready,
        "backend": getattr(npu_engine, "backend_type", "unknown"),
        "model_path": npu_engine.model_path,
        "labels_path": LOADED_LABELS_PATH,
        "class_count": len(COCO_CLASSES),
        "init_error": getattr(npu_engine, "init_error", None),
    }


async def _handle_detection(raw_bytes: bytes, client_ip: str = "client") -> Dict[str, Any]:
    """Shared pipeline: preprocess -> NPU inference -> YOLOv9 postprocess."""
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image payload received",
        )

    logger.info(f"Received detection request from {client_ip} ({len(raw_bytes) / 1024:.1f} KB)")

    t0 = time.perf_counter()

    # 1. Preprocess raw bytes into normalized NHWC tensor matching model dimensions
    in_w = getattr(npu_engine, "input_width", 320)
    in_h = getattr(npu_engine, "input_height", 320)

    try:
        tensor, orig_shape = preprocess_image(raw_bytes, target_w=in_w, target_h=in_h)
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to preprocess image: {str(e)}",
        )

    t1 = time.perf_counter()

    # 2. Execute inference via Torq NPU accelerator (IREE / SyNAP)
    try:
        raw_outputs = npu_engine.infer(tensor)
    except Exception as e:
        logger.error(f"Torq NPU inference failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NPU inference failed: {str(e)}",
        )

    t2 = time.perf_counter()

    # 3. Parse YOLOv9 output tensors into structured Frigate boundary boxes
    try:
        result = postprocess_yolov9(
            outputs=raw_outputs,
            orig_shape=orig_shape,
            score_threshold=SCORE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            input_width=in_w,
            input_height=in_h,
        )
    except Exception as e:
        logger.error(f"YOLOv9 postprocessing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse model output: {str(e)}",
        )

    t3 = time.perf_counter()

    preprocess_ms = (t1 - t0) * 1000.0
    inference_ms = (t2 - t1) * 1000.0
    postprocess_ms = (t3 - t2) * 1000.0
    total_ms = (t3 - t0) * 1000.0

    preds = result.get("predictions", [])
    if preds:
        det_summary = ", ".join(f"{p['label']} ({p['confidence']:.2f})" for p in preds)
    else:
        det_summary = "none"

    logger.info(
        f"Inference completed in {total_ms:.1f}ms "
        f"(pre={preprocess_ms:.1f}ms, npu={inference_ms:.1f}ms, post={postprocess_ms:.1f}ms) "
        f"| Detections ({len(preds)}): [{det_summary}]"
    )

    return {
        "success": True,
        "predictions": result["predictions"],
        "detections": result["detections"],
        "timing": {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "total_ms": round(total_ms, 2),
        },
    }


@app.post("/detect")
async def detect_raw_bytes(request: Request) -> Dict[str, Any]:
    """Primary detector endpoint accepting raw image bytes directly from Frigate."""
    content_type = request.headers.get("content-type", "")
    client_ip = request.client.host if request.client else "unknown"

    if "multipart/form-data" in content_type:
        form = await request.form()
        image_field = form.get("image") or form.get("file")
        if image_field is None:
            # Fallback to the first form entry
            for value in form.values():
                if hasattr(value, "read"):
                    image_field = value
                    break
        if image_field is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image uploaded in form data",
            )
        raw_bytes = await image_field.read()
    else:
        # Direct raw image stream in HTTP POST body
        raw_bytes = await request.body()

    return await _handle_detection(raw_bytes, client_ip=client_ip)


@app.post("/v1/vision/detection")
async def detect_vision_api(request: Request) -> Dict[str, Any]:
    """Compatible endpoint with DeepStack/CodeProject.AI detector configurations in Frigate."""
    return await detect_raw_bytes(request)


if __name__ == "__main__":
    try:
        uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
    except SystemExit as e:
        sys.exit(e.code)
