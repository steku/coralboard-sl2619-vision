"""Torq NPU / IREE Runtime Session for Coralboard SL2619."""

import logging
import os
from pathlib import Path
import sys
import tarfile
from typing import Any, List, Optional, Tuple
import zipfile
import numpy as np

logger = logging.getLogger("coral_vision.engine")

BASE_DIR = Path(__file__).resolve().parent


def _resolve_npu_backend() -> Tuple[Optional[Any], str, List[str]]:
    """Resolve and import Torq / IREE / SyNAP runtime backend.
    
    Returns:
        (runner_class_or_callable, backend_type, list_of_import_errors)
    """
    errors: List[str] = []

    # 1. Try direct import of torq.runtime
    try:
        from torq.runtime import VMFBInferenceRunner
        return VMFBInferenceRunner, "torq.runtime", errors
    except Exception as e:
        errors.append(f"torq.runtime: {type(e).__name__}: {e}")

    # 2. Try torq_runtime (flat module name from wheel)
    try:
        from torq_runtime import VMFBInferenceRunner
        return VMFBInferenceRunner, "torq_runtime", errors
    except Exception as e:
        errors.append(f"torq_runtime: {type(e).__name__}: {e}")

    # 3. Add system and user site-packages to sys.path
    candidate_paths = [
        "/usr/lib/python3.12/site-packages",
        "/usr/local/lib/python3.12/site-packages",
        os.path.expanduser("~/.local/lib/python3.12/site-packages"),
        "/usr/lib/python3/dist-packages",
        "/usr/local/lib/python3/dist-packages",
    ]
    for p in candidate_paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)

    # 4. Retry torq.runtime and torq_runtime after expanding sys.path
    try:
        from torq.runtime import VMFBInferenceRunner
        return VMFBInferenceRunner, "torq.runtime", errors
    except Exception as e:
        errors.append(f"torq.runtime (system paths): {type(e).__name__}: {e}")

    try:
        from torq_runtime import VMFBInferenceRunner
        return VMFBInferenceRunner, "torq_runtime", errors
    except Exception as e:
        errors.append(f"torq_runtime (system paths): {type(e).__name__}: {e}")

    # 5. Try pure iree.runtime
    try:
        from iree.runtime import VMFBInferenceRunner
        return VMFBInferenceRunner, "iree.runtime", errors
    except Exception as e:
        errors.append(f"iree.runtime: {type(e).__name__}: {e}")

    # 6. Try native Synaptics SyNAP SDK (for .synap models)
    try:
        from synap import Network
        return Network, "synap", errors
    except Exception as e:
        errors.append(f"synap: {type(e).__name__}: {e}")

    return None, "none", errors


def _detect_file_format(file_path: Path) -> str:
    """Detect binary format by reading magic bytes."""
    with open(file_path, "rb") as f:
        header = f.read(128)

    # IREE VMFB modules: either flatbytecode or ZIP container wrapping module.fb
    if file_path.suffix == ".vmfb" or b"module.fb" in header or b"IREE" in header[:32] or b"VMFB" in header[:32]:
        return "vmfb"
    if file_path.suffix == ".synap" or b"synap" in header[:32].lower() or b"SYNAP" in header[:32]:
        return "synap"
    # Hailo executable format (.hef) magic: 0x01 'H' 'E' 'F'
    if header.startswith(b"\x01HEF") or header.startswith(b"HEF"):
        return "hef"
    # ONNX protobuf magic: field 1 varint (0x08), version 7/8/9 (0x07-0x0a), field 2 length-delimited (0x12)
    if header.startswith(b"\x08") and len(header) > 3 and header[2] == 0x12:
        return "onnx"
    if b"ONNX" in header[:32]:
        return "onnx"
    if header.startswith(b"\x1f\x8b") or tarfile.is_tarfile(file_path):
        return "tar"
    if header.startswith(b"PK\x03\x04") or zipfile.is_zipfile(file_path):
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                if "module.fb" in zf.namelist():
                    return "vmfb"
        except Exception:
            pass
        return "zip"
    if header.startswith(b"<?xml") or b"<Error>" in header:
        return "xml_error"
    if header.startswith(b"{") and (b"error" in header.lower() or b"message" in header.lower()):
        return "json_error"

    if file_path.suffix == ".onnx":
        return "onnx"
    if file_path.suffix == ".synap":
        return "synap"
    return "vmfb"


class TorqVisionEngine:
    """Inference Engine strictly for Synaptics Torq NPU and SyNAP hardware accelerators."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.runner = None
        self.backend_type: str = "unknown"
        self._is_ready = False
        self.init_error: Optional[str] = None

    @property
    def input_width(self) -> int:
        if self.runner and hasattr(self.runner, "input_width"):
            return self.runner.input_width
        return 320

    @property
    def input_height(self) -> int:
        if self.runner and hasattr(self.runner, "input_height"):
            return self.runner.input_height
        return 320

    def _resolve_model_path(self) -> Path:
        """Resolve model path against workspace directory without auto-discovery."""
        target = Path(self.model_path)
        if not target.is_absolute():
            target = BASE_DIR / target

        if not target.exists():
            raise FileNotFoundError(
                f"Model file not found at '{target}'. "
                "The server will not search for alternative models; please check your config.yaml."
            )

        if not target.is_file():
            raise ValueError(f"Specified model path '{target}' is not a regular file.")

        return target

    def load(self) -> None:
        """Initialize the runtime session and load the model on the Torq NPU."""
        self._is_ready = False
        self.init_error = None

        # 1. Resolve and validate model path strictly as configured
        try:
            resolved_path = self._resolve_model_path()
            self.model_path = str(resolved_path)
        except Exception as e:
            self.init_error = f"Model validation failed: {e}"
            logger.error(self.init_error)
            raise

        if os.path.getsize(self.model_path) == 0:
            self.init_error = f"Model file '{self.model_path}' is completely empty (0 bytes)."
            logger.error(self.init_error)
            raise ValueError(self.init_error)

        file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        fmt = _detect_file_format(Path(self.model_path))
        logger.info(f"Loading model ({fmt.upper()}) from: {self.model_path} ({file_size_mb:.2f} MB)")

        # 2. Select backend based on detected format - NPU only, NEVER CPU
        try:
            if fmt == "hef":
                raise TypeError(
                    f"Model '{self.model_path}' is a Hailo-8/8L binary artifact (.hef) with magic header '\\x01HEF'. "
                    "The Coralboard SL2619 uses a Synaptics Torq NPU, which cannot execute Hailo models. "
                    "Please download or compile a Synaptics/Torq model (.vmfb / .synap) for the NPU."
                )
            elif fmt == "onnx" or self.model_path.endswith(".onnx"):
                raise RuntimeError(
                    f"Model '{self.model_path}' is an uncompiled ONNX model.\n"
                    "Inference must strictly execute on the Torq NPU; CPU fallback execution is prohibited.\n"
                    "ONNX models cannot run directly on the Torq NPU without compilation.\n"
                    "Please provide a compiled Torq VMFB (.vmfb) or SyNAP (.synap) model."
                )
            elif fmt == "synap" or self.model_path.endswith(".synap"):
                self.backend_type = "synap"
                from synap import Network
                self.runner = Network(self.model_path)
            else:
                # VMFB Bytecode on Torq NPU
                runner_cls, backend, import_errors = _resolve_npu_backend()
                self.backend_type = backend

                if runner_cls is None:
                    err_details = "\n  - " + "\n  - ".join(import_errors)
                    raise ImportError(
                        f"No Torq / SyNAP runtime available in Python {sys.version.split()[0]}!\n"
                        f"Import attempts failed with:{err_details}\n\n"
                        "To fix on Coralboard SL2619, install the Torq runtime wheel:\n"
                        "  pip install https://github.com/synaptics-torq/torq-compiler/releases/download/v2.1.0/torq_runtime-2.1.0-cp312-cp312-manylinux_2_28_aarch64.whl"
                    )

                logger.info(f"Initializing Torq VMFBInferenceRunner ({backend}): {self.model_path}")
                # Try initializing with default entrypoint ('main'), and fall back to 'main_graph' if needed
                init_attempts = [
                    {},
                    {"function": "main_graph"},
                    {"device_uri": "torq"},
                    {"device_uri": "torq", "function": "main_graph"},
                ]
                last_err = None
                for kwargs in init_attempts:
                    try:
                        self.runner = runner_cls(self.model_path, **kwargs)
                        break
                    except (TypeError, ValueError) as err:
                        last_err = err
                else:
                    if last_err:
                        raise last_err

            self._is_ready = True
            self.init_error = None
            logger.info(f"Vision inference session ({self.backend_type}) loaded successfully: {self.model_path}")
        except Exception as e:
            self.init_error = (
                f"Failed to initialize runner ({self.backend_type}) for '{self.model_path}': "
                f"{type(e).__name__}: {e}"
            )
            logger.error(self.init_error, exc_info=True)
            raise

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Execute inference on the Torq NPU.
        
        Args:
            input_tensor: Normalized (1, 320, 320, 3) NHWC tensor.
            
        Returns:
            List of output numpy arrays from the model head.
        """
        # Lazy recovery attempt if not initialized
        if not self._is_ready or self.runner is None:
            try:
                self.load()
            except Exception:
                pass

        if not self._is_ready or self.runner is None:
            raise RuntimeError(
                f"NPU engine is not initialized. Model '{self.model_path}' is not loaded. "
                f"Root cause: {self.init_error or 'Unknown initialization failure'}"
            )

        # Helper forward invocation
        def _run_forward(t: np.ndarray):
            if self.backend_type == "synap" and hasattr(self.runner, "predict"):
                return self.runner.predict([t])
            return self.runner.infer([t])

        # Proactively check model's expected layout (NCHW vs NHWC) and dtype (INT8 vs Float32)
        inp = input_tensor
        if hasattr(self.runner, "inputs_info") and self.runner.inputs_info:
            try:
                exp_info = self.runner.inputs_info[0]
                exp_shape = exp_info.shape
                if len(exp_shape) == 4 and exp_shape[1] in (1, 3, 4) and inp.ndim == 4 and inp.shape[-1] in (1, 3, 4):
                    inp = np.transpose(inp, (0, 3, 1, 2))
                if getattr(exp_info, "dtype", None) == np.int8 and inp.dtype != np.int8:
                    inp = np.clip(np.round(inp * 255.0) - 128.0, -128, 127).astype(np.int8)
            except Exception:
                pass

        # Execute inference through appropriate runtime (support float32/int8 and NHWC/NCHW formats)
        try:
            outputs = _run_forward(inp)
        except (TypeError, ValueError) as err:
            err_msg = str(err).lower()
            # 1. Try NCHW transposition if shape mismatch occurs (e.g. YOLO models exported from ONNX)
            if ("shape" in err_msg or "dimension" in err_msg or "size" in err_msg or "mismatch" in err_msg) and inp.ndim == 4 and inp.shape[-1] in (1, 3, 4):
                try:
                    inp_nchw = np.transpose(inp, (0, 3, 1, 2))
                    outputs = _run_forward(inp_nchw)
                    err = None
                    inp = inp_nchw
                except Exception as shape_err:
                    err = shape_err

            # 2. Try INT8 quantization if dtype error occurs
            if err is not None:
                err_msg = str(err).lower()
                if "int8" in err_msg or "i8" in err_msg or "dtype" in err_msg or "type" in err_msg:
                    inp_int8 = np.clip(np.round(inp * 255.0) - 128.0, -128, 127).astype(np.int8)
                    try:
                        outputs = _run_forward(inp_int8)
                    except (TypeError, ValueError):
                        if inp_int8.ndim == 4 and inp_int8.shape[-1] in (1, 3, 4):
                            inp_int8_nchw = np.transpose(inp_int8, (0, 3, 1, 2))
                            outputs = _run_forward(inp_int8_nchw)
                        else:
                            raise
                else:
                    raise

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        return [np.asarray(out) for out in outputs]

