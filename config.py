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

# COCO 80 Class Names
COCO_CLASSES = [
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
