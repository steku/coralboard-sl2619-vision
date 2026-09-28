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



