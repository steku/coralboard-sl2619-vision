# Synaptics Coralboard SL2619 Vision Proxy for Frigate NVR

Lightweight network proxy service running on the **Synaptics Coralboard SL2619** (Torq NPU, Python 3.12.9) bridging upstream **Frigate NVR** instances over HTTP without modifying Frigate's codebase.

## Features
- **Raw Byte Ingestion**: Accepts raw HTTP POST streams (`application/octet-stream`, `image/jpeg`) or multipart form payloads directly from Frigate.
- **Hardware-Targeted Preprocessing**: Transforms images into normalized $(1 \times 320 \times 320 \times 3)$ NHWC float32 tensors.
- **Torq NPU Hardware Acceleration**: Executes inference using the IREE Runtime session (`torq.runtime.VMFBInferenceRunner`).
- **YOLOv9 Output Parser**: Performs NMS and formats detections into Frigate's structured JSON bounding box schema.

## Project Structure
- [`server.py`](server.py): FastAPI application serving `/detect` and `/v1/vision/detection`.
- [`engine.py`](engine.py): IREE / Torq Runtime session management with automatic entrypoint and NCHW/NHWC layout detection.
- [`yolo.py`](yolo.py): Preprocessing (raw bytes $\to 320 \times 320 \times 3$) and postprocessing (YOLOv9 tensors $\to$ NMS $\to$ JSON).
- [`config.example.yaml`](config.example.yaml): Central user configuration template (copy to `config.yaml` to customize model, labels, thresholds, and server settings).
- [`config.py`](config.py): Configuration loader and runtime defaults with automatic label discovery.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): Full architectural overview and developer specification.

## Quick Start on Coralboard SL2619

```bash
# 1. Install dependencies (Python 3.12.9)
pip install -r requirements.txt

# 2. Select / Deploy Model

# Option A: Custom Trained Model (e.g. from frigate-model-training)
# Copy your compiled yolov9_320.vmfb and labels.json into models/:
mkdir -p models
cp /path/to/yolov9_320.vmfb models/yolov9_320.vmfb
cp /path/to/labels.json models/labels.json
# (Optional) Customize settings in config.yaml:
# cp config.example.yaml config.yaml


# Option B: Download official pre-compiled Synaptics Torq NPU base model (320x320 INT8 VMFB)
python3 download_model.py --synaptics-npu

# 3. Test detection locally
python3 test_detection.py --model models/yolov9_320.vmfb

# 4. Start the proxy server
python3 server.py
```

## Configuration (`config.yaml`)

The proxy service uses **`config.yaml`** as the single source of truth for user settings.

To customize settings, copy the template:
```bash
cp config.example.yaml config.yaml
```

### Configuration Schema

```yaml
# Server Network Settings
server:
  host: "0.0.0.0"
  port: 5000

# Model and Labels Configuration
model:
  path: "models/yolov9_320.vmfb"
  labels: "models/labels.json"  # Configurable file name (.json or .txt)
  # width: 320
  # height: 320

# Inference & Postprocessing Thresholds
detection:
  score_threshold: 0.20   # Minimum confidence threshold (0.0 to 1.0)
  iou_threshold: 0.40     # Non-Maximum Suppression (NMS) IoU threshold
  max_detections: 20      # Maximum detections per frame
  confidence_scale: 2.0   # Scale factor for INT8 quantized scores

# Frigate+ Model Downloader Settings (download_model.py)
frigate_plus:
  api_key: ""
  model_id: ""
  resolution: 640
  output_dir: "models"
```

### Custom Labels (`labels.json` / `labels.txt`)

Labels are dynamically discovered and mapped to model output classes. You can configure a custom label file name under `model.labels` in `config.yaml`, or place `labels.json` in the `models/` directory.

The parser automatically supports:
- **Ultralytics YOLO metadata**: `{"names": {"0": "person", "1": "car"}}` or `{"names": ["person", "car"]}`
- **Plain JSON list**: `["person", "car", "dog"]`
- **Direct index mapping**: `{"0": "person", "1": "car"}`
- **Frigate / Roboflow metadata**: `{"labels": [...]}` or `{"classes": [...]}`
- **Plain text file (`labels.txt`)**: One class label per line.

*(If no custom label file is found, it automatically defaults to the standard 80 COCO classes).*

## Frigate Configuration (Upstream Server)


When Frigate runs on an upstream machine, it communicates with the Coralboard SL2619 proxy service over HTTP.

### 1. Minimal Working Frigate `config.yml`

```yaml
detectors:
  coral_torq:
    type: deepstack
    api_url: http://<CORALBOARD_IP>:5000/v1/vision/detection
    api_timeout: 0.25  # 250ms timeout window

# Optional: configure input resolution for Frigate's tracker without a model path
model:
  width: 320
  height: 320

cameras:
  front_door:
    ffmpeg:
      inputs:
        - path: rtsp://...
          roles:
            - detect
    detect:
      enabled: True
      width: 1280
      height: 720
      fps: 5
```

---

### 2. Resolving: `Model does not support detector type of deepstack`

If Frigate's configuration editor reports:
```text
Your configuration is invalid.
Line Unknown: - Value error, Model does not support detector type of deepstack
```

#### Why This Happens
In Frigate's validation logic ([`detector_config.py`](https://github.com/blakeblackshear/frigate/blob/dev/frigate/detectors/detector_config.py)):
```python
if detector and detector not in model_info["supportedDetectors"]:
    raise ValueError(f"Model does not support detector type of {detector}")
```
Whenever a `path: plus://<model_id>` entry is configured, Frigate checks if the configured detector type matches the model's metadata. Because Frigate+ models declare hardware targets (`["synaptics"]`, `["edgetpu"]`, etc.) and never `["deepstack"]`, Frigate rejects the configuration.

#### How to Fix It
1. **Remove all `path: plus://...` entries** from your Frigate `config.yml`:
   - Under `detectors.coral_torq.model`
   - Under top-level `model.path`
   - Under `models:` lists (Frigate 0.16+)
   - Under any per-camera `model.path`
2. **Remember**: Frigate does **not** load or download the model when using an HTTP proxy. The model runs exclusively on the Coralboard SL2619 (downloaded via `download_model.py` and served by `server.py`). Frigate only sends HTTP POST requests to `api_url`.



