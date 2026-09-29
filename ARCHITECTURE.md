# Architecture Overview: Synaptics Coralboard SL2619 Vision Proxy for Frigate NVR

## 1. System Topology

To bridge the **Synaptics Coralboard SL2619** with an upstream **Frigate NVR** server without modifying Frigate's codebase, this proxy service provides a high-throughput, low-latency HTTP detector endpoint.

```
┌─────────────────────────────────┐
│       Frigate NVR Server        │
│  (HTTP / DeepStack Detector)    │
└────────────────┬────────────────┘
                 │
                 │ HTTP POST Raw Bytes (/detect or /v1/vision/detection)
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│        Synaptics Coralboard SL2619 (FastAPI Proxy)              │
│                                                                 │
│  1. Ingestion: Receive raw image bytes / stream                 │
│  2. Preprocess: Resize & normalize to (1, 320, 320, 3) NHWC     │
│  3. IREE Runtime: Dispatch to VMFBInferenceRunner session       │
│  4. Postprocess: YOLOv9 head decode, NMS, Frigate JSON format   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Hardware Dispatch (Direct DMA / Memory Map)
                 ▼
┌─────────────────────────────────┐
│       Synaptics Torq NPU        │
│    (Kelvin ML Core / IREE)      │
└─────────────────────────────────┘
```

---

## 2. Processing Pipeline

### Step 1: Accept Raw Image Bytes via HTTP POST
- **Endpoints**: `POST /detect` and `POST /v1/vision/detection`
- **Request Payloads**:
  - `application/octet-stream` or `image/jpeg`: Raw binary body passed directly from Frigate.
  - `multipart/form-data`: Standard form field `image` or `file` as used by HTTP detector clients.

### Step 2: Preprocess into Normalized Tensor `(1, 320, 320, 3)`
- Ingests compressed byte stream (JPEG, PNG) using `cv2.imdecode` or uncompressed byte buffers.
- Converts color space from BGR to RGB.
- Resizes to the model input dimensions: **$320 \times 320 \times 3$**.
- Normalizes pixel values from $[0, 255]$ uint8 to $[0.0, 1.0]$ float32.
- Expands dimensions to NHWC format: `(1, 320, 320, 3)`.

### Step 3: Inference via IREE Runtime for Synaptics Torq NPU
- Uses the compiled `.vmfb` artifact (`models/yolov9_320.vmfb`).
- Interfaced via `torq.runtime.VMFBInferenceRunner` (or `iree.runtime` fallback).
- Executes single-pass forward inference directly on the Torq NPU accelerator.

### Step 4: YOLOv9 Tensor Parsing & Frigate Structured JSON
- Receives the raw prediction tensor $(1, 84, N)$ or $(N, 84)$:
  - Columns 0–3: Bounding box parameters $(c_x, c_y, w, h)$.
  - Columns 4+: Confidence scores for the classes resolved via `config.yaml` / `labels.json` (e.g. 4 classes for custom dataset, or 80 for COCO defaults).
- Converts center-size coordinates to normalized corner coordinates $[y_{min}, x_{min}, y_{max}, x_{max}] \in [0.0, 1.0]$.
- Applies Non-Maximum Suppression (NMS) with `cv2.dnn.NMSBoxes` using configurable score and IoU thresholds (defaults: 0.20 score, 0.40 IoU).
- Emits dual compatible structures:
  1. `predictions`: Standard DeepStack/CodeProject array of labeled bounding boxes.
  2. `detections`: Frigate detector array `[class_id, score, ymin, xmin, ymax, xmax]`.


---

## 3. Data Contract: HTTP Response Schema

```json
{
  "success": true,
  "predictions": [
    {
      "label": "person",
      "confidence": 0.9125,
      "y_min": 0.1523,
      "x_min": 0.3845,
      "y_max": 0.7612,
      "x_max": 0.6214
    }
  ],
  "detections": [
    [0, 0.9125, 0.1523, 0.3845, 0.7612, 0.6214]
  ],
  "timing": {
    "preprocess_ms": 3.42,
    "inference_ms": 14.15,
    "postprocess_ms": 1.28,
    "total_ms": 18.85
  }
}
```

---

## 4. Frigate Configuration Integration

Add the Coralboard SL2619 detector to your Frigate `config.yml` without altering any Frigate code:

```yaml
detectors:
  coral_torq:
    type: deepstack
    api_url: http://<CORALBOARD_IP>:5000/v1/vision/detection
    api_timeout: 0.25 # 250ms timeout window

cameras:
  front_door:
    ffmpeg:
      inputs:
        - path: rtsp://camera-stream...
          roles:
            - detect
    detect:
      enabled: True
      width: 1280
      height: 720
      fps: 5
```

> [!IMPORTANT]
> **Do not define `model: path: plus://<id>` in Frigate:**
> Frigate attempts to validate `plus://` models against local detector plugins (`synaptics`, `edgetpu`), which causes `Value error, Model does not support detector type of deepstack`.
> Because the Coralboard runs the model externally over HTTP, the model is downloaded and loaded exclusively on the Coralboard via `download_model.py`.


---

## 5. Deployment on Coralboard SL2619 (Python 3.12.9)

### Systemd Service (`/etc/systemd/system/coral-vision.service`)

```ini
[Unit]
Description=Synaptics Coralboard SL2619 Vision Proxy for Frigate
After=network.target

[Service]
Type=simple
User=steve
WorkingDirectory=/home/steve/coral-sl2619-vision
ExecStart=/usr/bin/python3 -m uvicorn server:app --host 0.0.0.0 --port 5000 --workers 1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```
