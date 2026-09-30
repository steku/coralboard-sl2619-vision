"""Diagnostic CLI script to test NPU inference on a local image file."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from config import (
    COCO_CLASSES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MODEL_PATH,
    OUTPUT_SCALE,
    OUTPUT_ZERO_POINT,
    SCORE_THRESHOLD,
)
from engine import TorqVisionEngine
from yolo import postprocess_yolov9, preprocess_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_detection")

SAMPLE_URL = "https://huggingface.co/Synaptics/yolov26n_od/resolve/main/samples/dog_bike_car.jpg"


def get_test_image(image_path: Path) -> Path:
    """Ensure a valid test image is available."""
    if image_path.exists():
        return image_path

    # Try downloading sample image if not provided
    image_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading sample test image to {image_path}...")
    import urllib.request
    req = urllib.request.Request(SAMPLE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(image_path, "wb") as f:
        f.write(resp.read())
    logger.info("Sample image downloaded.")
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Torq NPU inference directly on an image.")
    parser.add_argument("--image", default="samples/dog_bike_car.jpg", help="Path to image file")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .vmfb model")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD, help="Confidence threshold")
    args = parser.parse_args()

    image_path = get_test_image(Path(args.image))

    logger.info(f"Loading Torq NPU model from: {args.model}")
    engine = TorqVisionEngine(model_path=args.model)
    engine.load()

    logger.info(f"Reading test image: {image_path}")
    with open(image_path, "rb") as f:
        raw_bytes = f.read()

    tensor, orig_shape = preprocess_image(raw_bytes, target_w=engine.input_width, target_h=engine.input_height)
    logger.info(f"Input tensor: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min():.2f}, max={tensor.max():.2f}")

    logger.info("Running NPU inference...")
    t0 = time.perf_counter()
    raw_outputs = engine.infer(tensor)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    output = raw_outputs[0]
    logger.info(f"Inference latency: {infer_ms:.2f}ms")
    logger.info(f"Raw output tensor: shape={output.shape}, dtype={output.dtype}, min={output.min()}, max={output.max()}")

    # Postprocessing
    result = postprocess_yolov9(
        outputs=raw_outputs,
        orig_shape=orig_shape,
        score_threshold=args.threshold,
        input_width=engine.input_width,
        input_height=engine.input_height,
    )

    preds = result.get("predictions", [])
    logger.info(f"Detections found ({len(preds)}) at threshold {args.threshold}:")
    for p in preds:
        box = f"[{p['x_min']}, {p['y_min']}, {p['x_max']}, {p['y_max']}]"
        logger.info(f"  -> {p['label']:<15} confidence: {p['confidence']:.3f} | box: {box}")

    if not preds:
        # Diagnostic: print top 5 candidates regardless of threshold
        raw_arr = np.asarray(output)
        if raw_arr.dtype == np.int8:
            raw_arr = (raw_arr.astype(np.float32) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE
        if raw_arr.ndim == 3:
            raw_arr = np.squeeze(raw_arr, axis=0)
        if raw_arr.ndim == 2 and raw_arr.shape[0] < raw_arr.shape[1]:
            raw_arr = raw_arr.T
        scores = np.max(raw_arr[:, 4:], axis=1)
        top5_idx = np.argsort(scores)[::-1][:5]
        logger.info("Top 5 model candidates (unfiltered):")
        for idx in top5_idx:
            cid = int(np.argmax(raw_arr[idx, 4:]))
            label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else str(cid)
            logger.info(f"  candidate: {label:<15} score={scores[idx]:.4f}")


if __name__ == "__main__":
    main()
