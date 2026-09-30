"""Image preprocessing and YOLOv9 tensor postprocessing for Coralboard SL2619."""

from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from config import (
    COCO_CLASSES,
    CONFIDENCE_SCALE,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    OUTPUT_SCALE,
    OUTPUT_ZERO_POINT,
    SCORE_THRESHOLD,
)


def preprocess_image(raw_bytes: bytes, target_w: int = INPUT_WIDTH, target_h: int = INPUT_HEIGHT) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Decode raw image bytes into a letterboxed, normalized (1, H, W, 3) NHWC float32 tensor.
    
    Preserves aspect ratio by scaling and centering with standard 114 gray padding.

    Returns:
        tensor: np.ndarray of shape (1, target_h, target_w, 3) normalized to [0.0, 1.0].
        orig_shape: (original_height, original_width).
    """
    # 1. Decode compressed bytes (JPEG, PNG, etc.) via OpenCV
    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    # 2. Fallback: handle raw uncompressed byte stream if imdecode fails
    if image is None:
        expected_size = target_h * target_w * 3
        if len(raw_bytes) == expected_size:
            image = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((target_h, target_w, 3))
        else:
            raise ValueError(f"Failed to decode image bytes (length: {len(raw_bytes)})")

    orig_h, orig_w = image.shape[:2]

    # 3. Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 4. Letterbox resize to preserve aspect ratio (pad with 114 gray)
    if (orig_w, orig_h) == (target_w, target_h):
        letterboxed = image_rgb
    else:
        scale = min(float(target_w) / float(orig_w), float(target_h) / float(orig_h))
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        if (orig_w, orig_h) != (new_w, new_h):
            resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image_rgb

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        letterboxed = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        letterboxed[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # 5. Normalize uint8 [0, 255] to float32 [0.0, 1.0] and add batch dimension -> (1, target_h, target_w, 3)
    tensor = letterboxed.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, (orig_h, orig_w)


def postprocess_yolov9(
    outputs: Any,
    orig_shape: Tuple[int, int] = (INPUT_HEIGHT, INPUT_WIDTH),
    score_threshold: float = SCORE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
    input_width: int = INPUT_WIDTH,
    input_height: int = INPUT_HEIGHT,
    out_scale: float = OUTPUT_SCALE,
    out_zp: float = OUTPUT_ZERO_POINT,
) -> Dict[str, Any]:
    """Parse YOLOv9 output tensors into Frigate-compatible bounding box structures.
    
    Args:
        outputs: Output array or list of arrays from IREE Runtime inference.
        orig_shape: (original_height, original_width) of the incoming image.
        score_threshold: Minimum confidence score to retain detection.
        iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
        max_detections: Maximum detections to return (Frigate standard is 20).
        input_width: Model tensor input width.
        input_height: Model tensor input height.
        out_scale: INT8 output dequantization scale factor.
        out_zp: INT8 output dequantization zero point.
        
    Returns:
        Dictionary containing:
        - 'predictions': DeepStack-compatible list of objects with label, confidence, pixel coordinates.
        - 'detections': Frigate-native list of [class_id, score, ymin, xmin, ymax, xmax].
    """
    if isinstance(outputs, (list, tuple)):
        raw_pred = outputs[0]
    else:
        raw_pred = outputs

    # Ensure numpy array
    raw_pred = np.asarray(raw_pred)

    # Dequantize int8 predictions if model outputs INT8 (e.g. YOLO Torq NPU head)
    if raw_pred.dtype == np.int8:
        raw_pred = (raw_pred.astype(np.float32) - out_zp) * out_scale
    elif raw_pred.dtype != np.float32:
        raw_pred = raw_pred.astype(np.float32)

    # Squeeze batch dimension if present: (1, 84, N) -> (84, N) or (1, N, 84) -> (N, 84)
    if raw_pred.ndim == 3:
        raw_pred = np.squeeze(raw_pred, axis=0)

    # YOLO head standard format: (channels, num_anchors) e.g. (84, 2100)
    # Transpose if necessary to get (num_anchors, 84)
    if raw_pred.ndim == 2 and raw_pred.shape[0] < raw_pred.shape[1]:
        raw_pred = raw_pred.T

    # Bounding boxes (cx, cy, w, h) are the first 4 columns, followed by 80 COCO class scores
    boxes_raw = raw_pred[:, :4]
    class_scores = raw_pred[:, 4:]

    # Class ID and highest score per candidate
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    # Score filtering
    valid_mask = scores >= score_threshold
    if not np.any(valid_mask):
        return {"predictions": [], "detections": []}

    boxes_raw = boxes_raw[valid_mask]
    scores = scores[valid_mask]
    class_ids = class_ids[valid_mask]

    # Convert cx, cy, w, h to xyxy normalized [0.0, 1.0] relative to model input dimensions
    # Supports both normalized coordinates [0.0, 1.0] and pixel space coordinates [0, 320]
    if len(boxes_raw) > 0 and np.max(boxes_raw) <= 1.05:
        cx = boxes_raw[:, 0] * float(input_width)
        cy = boxes_raw[:, 1] * float(input_height)
        w = boxes_raw[:, 2] * float(input_width)
        h = boxes_raw[:, 3] * float(input_height)
    else:
        cx = boxes_raw[:, 0]
        cy = boxes_raw[:, 1]
        w = boxes_raw[:, 2]
        h = boxes_raw[:, 3]

    x1 = np.clip((cx - w / 2.0) / float(input_width), 0.0, 1.0)
    y1 = np.clip((cy - h / 2.0) / float(input_height), 0.0, 1.0)
    x2 = np.clip((cx + w / 2.0) / float(input_width), 0.0, 1.0)
    y2 = np.clip((cy + h / 2.0) / float(input_height), 0.0, 1.0)

    # Prepare boxes in [x, y, width, height] format in pixels for cv2.dnn.NMSBoxes
    nms_boxes = []
    for bx1, by1, bx2, by2 in zip(x1, y1, x2, y2):
        px = int(bx1 * input_width)
        py = int(by1 * input_height)
        pw = int(max(0.0, (bx2 - bx1) * input_width))
        ph = int(max(0.0, (by2 - by1) * input_height))
        nms_boxes.append([px, py, pw, ph])

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        scores.tolist(),
        score_threshold=float(score_threshold),
        nms_threshold=float(iou_threshold),
    )

    orig_h, orig_w = orig_shape
    in_w = float(input_width)
    in_h = float(input_height)

    # Calculate letterbox scale and padding for unletterbox coordinate transformation
    scale = min(in_w / float(orig_w), in_h / float(orig_h)) if orig_w > 0 and orig_h > 0 else 1.0
    pad_x = (in_w - float(orig_w) * scale) / 2.0
    pad_y = (in_h - float(orig_h) * scale) / 2.0

    predictions: List[Dict[str, Any]] = []
    detections: List[List[float]] = []

    if len(indices) > 0:
        flat_indices = np.array(indices).flatten()[:max_detections]
        for idx in flat_indices:
            cid = int(class_ids[idx])
            label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"
            # Scale INT8 saturated scores (0.0-0.5) to standard Frigate detector range (0.0-1.0)
            conf = min(1.0, float(scores[idx]) * CONFIDENCE_SCALE)

            norm_x1 = float(x1[idx])
            norm_y1 = float(y1[idx])
            norm_x2 = float(x2[idx])
            norm_y2 = float(y2[idx])

            # Unletterbox model coordinates (norm_x, norm_y in input_width x input_height space):
            orig_x1 = (norm_x1 * in_w - pad_x) / scale
            orig_y1 = (norm_y1 * in_h - pad_y) / scale
            orig_x2 = (norm_x2 * in_w - pad_x) / scale
            orig_y2 = (norm_y2 * in_h - pad_y) / scale

            # Clip coordinates to original image bounds
            orig_x1 = max(0.0, min(float(orig_w), orig_x1))
            orig_y1 = max(0.0, min(float(orig_h), orig_y1))
            orig_x2 = max(orig_x1, min(float(orig_w), orig_x2))
            orig_y2 = max(orig_y1, min(float(orig_h), orig_y2))

            # DeepStack returns pixel coordinates based on original image dimensions
            predictions.append({
                "label": label,
                "confidence": round(conf, 4),
                "y_min": int(round(orig_y1)),
                "x_min": int(round(orig_x1)),
                "y_max": int(round(orig_y2)),
                "x_max": int(round(orig_x2)),
            })

            # Frigate detection array: [class_id, confidence, ymin, xmin, ymax, xmax] normalized to [0.0, 1.0]
            norm_ymin = orig_y1 / float(orig_h) if orig_h > 0 else 0.0
            norm_xmin = orig_x1 / float(orig_w) if orig_w > 0 else 0.0
            norm_ymax = orig_y2 / float(orig_h) if orig_h > 0 else 0.0
            norm_xmax = orig_x2 / float(orig_w) if orig_w > 0 else 0.0

            detections.append([
                cid,
                round(conf, 4),
                round(float(norm_ymin), 4),
                round(float(norm_xmin), 4),
                round(float(norm_ymax), 4),
                round(float(norm_xmax), 4),
            ])

    return {
        "success": True,
        "predictions": predictions,
        "detections": detections,
    }
