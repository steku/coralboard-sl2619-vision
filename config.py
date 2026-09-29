import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent

# Model Configuration
MODEL_PATH = "models/yolov9_320.vmfb"
LABELS_PATH: Optional[str] = "models/labels.json"  # Path to labels JSON/txt file, or None for auto-discovery
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
INPUT_CHANNELS = 3

# Inference Thresholds
SCORE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.4
MAX_DETECTIONS = 20
CONFIDENCE_SCALE = 2.0  # Rescales INT8 saturated [0.0, 0.5] confidences to standard [0.0, 1.0] for Frigate

# Server Settings
HOST = "0.0.0.0"
PORT = 5000


def _load_yaml_config() -> Dict[str, Any]:
    """Read optional config.yaml or config.yml if present."""
    for cfg_name in ("config.yaml", "config.yml"):
        cfg_file = BASE_DIR / cfg_name
        if cfg_file.exists():
            try:
                import yaml
                with open(cfg_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logging.getLogger("coral_vision.config").warning(f"Failed to read {cfg_file}: {e}")
    return {}


# Apply overrides from config.yaml if present
_yaml_cfg = _load_yaml_config()
if _yaml_cfg:
    _server_cfg = _yaml_cfg.get("server", {})
    if isinstance(_server_cfg, dict):
        if "host" in _server_cfg:
            HOST = str(_server_cfg["host"])
        if "port" in _server_cfg:
            PORT = int(_server_cfg["port"])

    _model_cfg = _yaml_cfg.get("model", {})
    if isinstance(_model_cfg, dict):
        if "path" in _model_cfg:
            MODEL_PATH = str(_model_cfg["path"])
        if "labels" in _model_cfg:
            LABELS_PATH = str(_model_cfg["labels"])
        elif "labels_path" in _model_cfg:
            LABELS_PATH = str(_model_cfg["labels_path"])
        if "width" in _model_cfg:
            INPUT_WIDTH = int(_model_cfg["width"])
        if "height" in _model_cfg:
            INPUT_HEIGHT = int(_model_cfg["height"])

    _det_cfg = _yaml_cfg.get("detection", {})
    if isinstance(_det_cfg, dict):
        if "score_threshold" in _det_cfg:
            SCORE_THRESHOLD = float(_det_cfg["score_threshold"])
        if "iou_threshold" in _det_cfg:
            IOU_THRESHOLD = float(_det_cfg["iou_threshold"])
        if "max_detections" in _det_cfg:
            MAX_DETECTIONS = int(_det_cfg["max_detections"])
        if "confidence_scale" in _det_cfg:
            CONFIDENCE_SCALE = float(_det_cfg["confidence_scale"])

    if "labels_path" in _yaml_cfg:
        LABELS_PATH = str(_yaml_cfg["labels_path"])
    if "model_path" in _yaml_cfg:
        MODEL_PATH = str(_yaml_cfg["model_path"])

# Environment variable overrides
if os.environ.get("LABELS_PATH"):
    LABELS_PATH = os.environ["LABELS_PATH"]
if os.environ.get("MODEL_PATH"):
    MODEL_PATH = os.environ["MODEL_PATH"]

# COCO 80 Default Class Names
DEFAULT_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

logger = logging.getLogger("coral_vision.config")


def _parse_label_data(data: Any) -> Optional[List[str]]:
    """Extract list of class names from various JSON structures."""
    # 1. Plain list of class names: ["person", "car", ...]
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "name" in data[0]:
            return [str(item["name"]) for item in sorted(data, key=lambda x: int(x.get("id", 0)))]
        return [str(x) for x in data]

    if isinstance(data, dict):
        # 2. Wrapped under common keys: "names", "labels", "classes", "categories"
        for key in ("names", "labels", "classes", "categories"):
            if key in data:
                sub = data[key]
                if isinstance(sub, list):
                    if sub and isinstance(sub[0], dict) and "name" in sub[0]:
                        return [str(item["name"]) for item in sorted(sub, key=lambda x: int(x.get("id", 0)))]
                    return [str(x) for x in sub]
                elif isinstance(sub, dict):
                    sorted_items = sorted(
                        sub.items(),
                        key=lambda item: int(item[0]) if str(item[0]).isdigit() else item[0],
                    )
                    return [str(v) for _, v in sorted_items]

        # 3. Direct dictionary mapping: {"0": "person", "1": "car"} or {0: "person", 1: "car"}
        if data and all(str(k).isdigit() for k in data.keys()):
            sorted_items = sorted(data.items(), key=lambda item: int(item[0]))
            return [str(v) for _, v in sorted_items]

    return None


def _load_classes() -> List[str]:
    """Load model class labels from LABELS_PATH or standard locations, falling back to COCO defaults."""
    candidate_files = []

    # 1. Configured explicit labels path (from config.py, config.yaml, or env var)
    if LABELS_PATH:
        p = Path(LABELS_PATH)
        candidate_files.append(p if p.is_absolute() else (BASE_DIR / p))

    # 2. Standard auto-discovery paths
    model_dir = (BASE_DIR / MODEL_PATH).parent
    candidate_files.extend([
        model_dir / "labels.json",
        BASE_DIR / "models" / "labels.json",
        BASE_DIR / "labels.json",
        model_dir / "labels.txt",
        BASE_DIR / "models" / "labels.txt",
    ])

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for p in candidate_files:
        p_res = p.resolve()
        if p_res not in seen:
            seen.add(p_res)
            unique_candidates.append(p)

    for labels_file in unique_candidates:
        if labels_file.exists() and labels_file.is_file():
            try:
                if labels_file.suffix == ".json":
                    with open(labels_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    classes = _parse_label_data(data)
                    if classes:
                        logger.info(f"Loaded {len(classes)} classes from {labels_file.name}")
                        return classes
                elif labels_file.suffix == ".txt":
                    with open(labels_file, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        logger.info(f"Loaded {len(lines)} classes from {labels_file.name}")
                        return lines
            except Exception as e:
                logger.warning(f"Failed to load labels from {labels_file}: {e}")

    logger.info(f"Using default COCO classes ({len(DEFAULT_COCO_CLASSES)} classes)")
    return DEFAULT_COCO_CLASSES


COCO_CLASSES = _load_classes()

