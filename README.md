# Synaptics Coralboard SL2619 Vision Proxy for Frigate NVR

Lightweight network proxy service running on the **Synaptics Coralboard SL2619** (Torq NPU, Python 3.12.9) bridging upstream **Frigate NVR** instances over HTTP without modifying Frigate's codebase.

## Features
- **Raw Byte Ingestion**: Accepts raw HTTP POST streams (`application/octet-stream`, `image/jpeg`) or multipart form payloads directly from Frigate.
- **Hardware-Targeted Preprocessing**: Transforms images into normalized $(1 \times 320 \times 320 \times 3)$ NHWC float32 tensors.
- **Torq NPU Hardware Acceleration**: Executes inference using the IREE Runtime session (`torq.runtime.VMFBInferenceRunner`).
- **YOLOv9 Output Parser**: Performs NMS and formats detections into Frigate's structured JSON bounding box schema.

## Project Structure
- [`server.py`](file:///Users/stephen/Projects/coral-sl2619-vision/server.py): FastAPI application serving `/detect` and `/v1/vision/detection`.
- [`engine.py`](file:///Users/stephen/Projects/coral-sl2619-vision/engine.py): IREE Runtime session management for the Synaptics Torq NPU.
- [`yolo.py`](file:///Users/stephen/Projects/coral-sl2619-vision/yolo.py): Preprocessing (raw bytes $\to 320 \times 320 \times 3$) and postprocessing (YOLOv9 tensors $\to$ NMS $\to$ JSON).
- [`config.py`](file:///Users/stephen/Projects/coral-sl2619-vision/config.py): Server and inference parameters.
- [`ARCHITECTURE.md`](file:///Users/stephen/Projects/coral-sl2619-vision/ARCHITECTURE.md): Full architectural overview and developer specification.

## Quick Start on Coralboard SL2619

```bash
# 1. Install dependencies (Python 3.12.9)
pip install -r requirements.txt

# 2. Configure Frigate+ API key
cp config.example.yaml config.yaml
# Edit config.yaml and place your real Frigate+ API key

# 3. Download model from Frigate+
python3 download_model.py

# 4. Start the proxy server
python3 server.py
```

## Frigate Configuration (Upstream Server)

When Frigate runs on a separate machine, it offloads object detection to the Coralboard proxy over HTTP.

### 1. Frigate `config.yml`

```yaml
detectors:
  coral_torq:
    type: deepstack
    api_url: http://<CORALBOARD_IP>:5000/v1/vision/detection
    api_timeout: 0.25  # Timeout in seconds

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

### 2. Troubleshooting: `Model does not support detector type of deepstack`

If Frigate fails to start with:
```text
Line Unknown: - Value error, Model does not support detector type of deepstack
```

**Cause**: 
Your Frigate `config.yml` contains a `model:` section with `path: plus://<model_id>`. 
Frigate checks `plus://` models against built-in accelerator plugins (where the model metadata specifies `supportedDetectors: ["synaptics"]`). Because `deepstack` is not in that list, Frigate throws a validation error.

**Fix**:
Remove `path: plus://...` from Frigate's `config.yml`. When using an external HTTP detector (`type: deepstack`), Frigate does **not** load or execute the model. The model runs exclusively on the Coralboard SL2619, which downloads and loads it directly via `download_model.py`.


