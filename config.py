# Model Configuration
MODEL_PATH = "models/yolov9_320.vmfb"
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

# Model Classes (matching yolo_dataset/data.yaml)
COCO_CLASSES = [
    "person",
    "dog",
    "cat",
    "car",
]
