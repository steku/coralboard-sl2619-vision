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


def _inspect_and_unpack_if_archive(file_path: Path) -> Path:
    """Inspect downloaded artifact, auto-unpack if compressed, and return path to model."""
    if not file_path.exists():
        return file_path

    if file_path.stat().st_size == 0:
        raise ValueError(f"Model file '{file_path}' is completely empty (0 bytes).")

    # Read first 128 bytes to check magic headers
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

    # Check for GZIP / TAR archive
    if header.startswith(b"\x1f\x8b") or tarfile.is_tarfile(file_path):
        logger.info(f"Archive detected at {file_path}. Unpacking tarball...")
        try:
            with tarfile.open(file_path, "r:*") as tar:
                tar.extractall(path=out_dir)
            logger.info("Tar archive extracted successfully.")
        except Exception as e:
            logger.warning(f"Failed to extract as tar archive: {e}")

    # Check for ZIP archive
    elif header.startswith(b"PK\x03\x04") or zipfile.is_zipfile(file_path):
        logger.info(f"Zip archive detected at {file_path}. Unpacking...")
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(path=out_dir)
            logger.info("Zip archive extracted successfully.")
        except Exception as e:
            logger.warning(f"Failed to extract as zip archive: {e}")

    # Find the extracted model inside output dir
    for ext in (".vmfb", ".synap"):
        for extracted in out_dir.glob(f"*{ext}"):
            if extracted.name not in ("yolov9_320.vmfb", "model.vmfb") and extracted.is_file():
                return extracted

    return file_path


class TorqVisionEngine:
    """IREE Runtime Inference Engine for Synaptics Torq NPU."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.runner = None
        self.backend_type: str = "unknown"
        self._is_ready = False
        self.init_error: Optional[str] = None

    def _resolve_model_path(self) -> Path:
        """Resolve model path against workspace and perform auto-discovery."""
        target = Path(self.model_path)
        if not target.is_absolute():
            target = BASE_DIR / target

        if not target.exists():
            model_dir = target.parent if target.parent.exists() else (BASE_DIR / "models")
            if model_dir.is_dir():
                candidates = [
                    f for f in model_dir.iterdir()
                    if f.suffix in (".vmfb", ".synap") and f.is_file()
                ]
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    target = candidates[0]
                    logger.info(f"Auto-discovered model artifact: {target}")

        return target

    def load(self) -> None:
        """Initialize the NPU runtime session and load the compiled model."""
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
                f"Model file not found at '{self.model_path}' and no .vmfb files found in 'models/'. "
                "Please run 'python3 download_model.py' to download your model from Frigate+."
            )
            logger.error(self.init_error)
            raise FileNotFoundError(self.init_error)

        file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        logger.info(f"Loading Torq NPU model from: {self.model_path} ({file_size_mb:.2f} MB)")

        # 2. Resolve NPU backend runtime
        runner_cls, backend, import_errors = _resolve_npu_backend()
        self.backend_type = backend

        if runner_cls is None:
            err_details = "\n  - " + "\n  - ".join(import_errors)
            self.init_error = (
                f"No Torq / SyNAP runtime available in Python {sys.version.split()[0]}!\n"
                f"Import attempts failed with:{err_details}\n\n"
                "To fix on Coralboard SL2619, install the Torq runtime wheel:\n"
                "  pip install https://github.com/synaptics-torq/torq-compiler/releases/download/v2.1.0/torq_runtime-2.1.0-cp312-cp312-manylinux_2_28_aarch64.whl"
            )
            logger.error(self.init_error)
            raise ImportError(self.init_error)

        # 3. Instantiate NPU session
        try:
            if backend == "synap":
                logger.info(f"Initializing SyNAP Network: {self.model_path}")
                self.runner = runner_cls(self.model_path)
            else:
                logger.info(f"Initializing Torq VMFBInferenceRunner ({backend}): {self.model_path}")
                try:
                    self.runner = runner_cls(self.model_path)
                except TypeError:
                    self.runner = runner_cls(self.model_path, device_uri="torq")

            self._is_ready = True
            self.init_error = None
            logger.info(f"Synaptics NPU session ({backend}) loaded successfully with model: {self.model_path}")
        except Exception as e:
            self.init_error = (
                f"Failed to initialize NPU runner ({backend}) for '{self.model_path}': "
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

        # Execute inference through appropriate runtime
        if self.backend_type == "synap" and hasattr(self.runner, "predict"):
            outputs = self.runner.predict([input_tensor])
        else:
            outputs = self.runner.infer([input_tensor])

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        return [np.asarray(out) for out in outputs]

