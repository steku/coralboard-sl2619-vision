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


def _inspect_and_unpack_if_archive(file_path: Path) -> Path:
    """Inspect downloaded artifact, auto-unpack if compressed, and return path to model."""
    if not file_path.exists():
        return file_path

    if file_path.stat().st_size == 0:
        raise ValueError(f"Model file '{file_path}' is completely empty (0 bytes).")

    # Never extract .vmfb files (IREE modules are packaged as zip containers with module.fb)
    if file_path.suffix == ".vmfb" or _detect_file_format(file_path) == "vmfb":
        return file_path

    with open(file_path, "rb") as f:
        header = f.read(128)

    # Check for text/XML error response (e.g. S3 AccessDenied)
    if header.startswith(b"<?xml") or b"<Error>" in header:
        err_snippet = header.decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Model file '{file_path}' is an XML error document from storage, not a model binary:\n{err_snippet}"
        )

    if header.startswith(b"{") and (b"error" in header.lower() or b"message" in header.lower()):
        err_snippet = header.decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Model file '{file_path}' contains JSON error response:\n{err_snippet}"
        )

    out_dir = file_path.parent

    is_archive = False
    # Check for GZIP / TAR archive
    if header.startswith(b"\x1f\x8b") or tarfile.is_tarfile(file_path):
        logger.info(f"Archive detected at {file_path}. Unpacking tarball...")
        try:
            with tarfile.open(file_path, "r:*") as tar:
                tar.extractall(path=out_dir)
            logger.info("Tar archive extracted successfully.")
            is_archive = True
        except Exception as e:
            logger.warning(f"Failed to extract as tar archive: {e}")

    # Check for general ZIP archive (excluding IREE bytecode vmfb)
    elif header.startswith(b"PK\x03\x04") or zipfile.is_zipfile(file_path):
        is_vmfb = False
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                if "module.fb" in zf.namelist():
                    is_vmfb = True
        except Exception:
            pass

        if not is_vmfb:
            logger.info(f"Zip archive detected at {file_path}. Unpacking...")
            try:
                with zipfile.ZipFile(file_path, "r") as zf:
                    zf.extractall(path=out_dir)
                logger.info("Zip archive extracted successfully.")
                is_archive = True
            except Exception as e:
                logger.warning(f"Failed to extract as zip archive: {e}")

    # Check for Hailo HEF model misnamed as .vmfb
    if header.startswith(b"\x01HEF") or header.startswith(b"HEF"):
        hef_path = file_path.with_suffix(".hef")
        if hef_path != file_path:
            try:
                if not hef_path.exists():
                    file_path.rename(hef_path)
                file_path = hef_path
                logger.info(f"Renamed misnamed Hailo artifact: {hef_path.name}")
            except Exception as e:
                logger.warning(f"Could not rename {file_path} -> {hef_path}: {e}")
        return file_path

    # Check for ONNX model disguised as .vmfb
    if header.startswith(b"\x08") and len(header) > 3 and header[2] == 0x12:
        logger.info(f"Artifact '{file_path.name}' is an ONNX model (Protobuf wire format ir_version=7).")
        onnx_path = file_path.with_suffix(".onnx")
        if onnx_path != file_path:
            try:
                if not onnx_path.exists():
                    file_path.rename(onnx_path)
                file_path = onnx_path
            except Exception as e:
                logger.warning(f"Could not rename {file_path} -> {onnx_path}: {e}")

    # If an archive was extracted, find the newly extracted model inside output dir
    if is_archive:
        for ext in (".vmfb", ".synap", ".onnx"):
            for extracted in out_dir.glob(f"*{ext}"):
                if extracted.name not in ("yolov9_320.vmfb", "model.vmfb", "yolov9_320.onnx", "model.onnx") and extracted.is_file():
                    try:
                        if _detect_file_format(extracted) in ("vmfb", "synap", "onnx"):
                            return extracted
                    except Exception:
                        pass

    return file_path


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
        """Resolve model path against workspace and perform auto-discovery."""
        target = Path(self.model_path)
        if not target.is_absolute():
            target = BASE_DIR / target

        model_dir = target.parent if target.parent.exists() else (BASE_DIR / "models")
        if model_dir.is_dir():
            # Clean up and rename any misnamed Hailo .hef files ending in .vmfb
            for f in list(model_dir.iterdir()):
                if f.is_file() and f.suffix == ".vmfb":
                    try:
                        if _detect_file_format(f) == "hef":
                            hef_dest = f.with_suffix(".hef")
                            if not hef_dest.exists():
                                f.rename(hef_dest)
                                logger.info(f"Renamed misnamed Hailo model: {f.name} -> {hef_dest.name}")
                    except Exception:
                        pass

        # If configured path does not exist, points to Hailo (.hef), or points to ONNX, auto-discover real NPU binaries
        is_invalid = (
            not target.exists()
            or _detect_file_format(target) in ("hef", "onnx")
            or target.suffix == ".onnx"
        )
        if is_invalid:
            if model_dir.is_dir():
                candidates = [
                    f for f in model_dir.iterdir()
                    if f.suffix in (".vmfb", ".synap") and f.is_file()
                ]
                # Filter for genuine Torq/SyNAP NPU binaries
                compatible = []
                for c in candidates:
                    try:
                        c_fmt = _detect_file_format(c)
                        if c_fmt in ("vmfb", "synap"):
                            compatible.append(c)
                    except Exception:
                        pass

                if compatible:
                    compatible.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    target = compatible[0]
                    logger.info(f"Auto-discovered compatible NPU model artifact: {target}")
                else:
                    raise FileNotFoundError(
                        f"Configured model '{target}' is an uncompiled ONNX or incompatible model, "
                        "and no genuine Torq NPU models (.vmfb / .synap) were found in 'models/'.\n"
                        "To download and install the official hardware-accelerated model, run:\n"
                        "  python3 download_model.py --synaptics-npu"
                    )

        return target

    def load(self) -> None:
        """Initialize the runtime session and load the model on the Torq NPU."""
        self._is_ready = False
        self.init_error = None

        # 1. Resolve and validate model path
        try:
            resolved_path = self._resolve_model_path()
            if resolved_path.exists():
                resolved_path = _inspect_and_unpack_if_archive(resolved_path)
            self.model_path = str(resolved_path)
        except Exception as e:
            self.init_error = f"Model validation failed: {e}"
            logger.error(self.init_error, exc_info=True)
            raise

        if not os.path.exists(self.model_path):
            self.init_error = (
                f"Model file not found at '{self.model_path}' and no compatible NPU models found in 'models/'. "
                "Please run 'python3 download_model.py' to download your Torq NPU model (.vmfb / .synap)."
            )
            logger.error(self.init_error)
            raise FileNotFoundError(self.init_error)

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
                try:
                    self.runner = runner_cls(self.model_path)
                except TypeError:
                    self.runner = runner_cls(self.model_path, device_uri="torq")

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

        # Execute inference through appropriate runtime (support both float32 and int8 models)
        inp = input_tensor
        try:
            if self.backend_type == "synap" and hasattr(self.runner, "predict"):
                outputs = self.runner.predict([inp])
            else:
                outputs = self.runner.infer([inp])
        except (TypeError, ValueError) as err:
            err_msg = str(err).lower()
            if "int8" in err_msg or "i8" in err_msg or "dtype" in err_msg or "type" in err_msg:
                # Convert float32 [0.0, 1.0] -> int8 [-128, 127] with scale 1/255 and zero-point -128
                inp_int8 = np.clip(np.round(inp * 255.0) - 128.0, -128, 127).astype(np.int8)
                if self.backend_type == "synap" and hasattr(self.runner, "predict"):
                    outputs = self.runner.predict([inp_int8])
                else:
                    outputs = self.runner.infer([inp_int8])
            else:
                raise

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        return [np.asarray(out) for out in outputs]

