"""Torq NPU / IREE Runtime Session for Coralboard SL2619."""

import logging
import os
import sys
from typing import Any, List, Optional
import numpy as np

logger = logging.getLogger("coral_vision.engine")


def _import_vmfb_runner():
    """Import VMFBInferenceRunner supporting torq.runtime, torq_runtime, or iree.runtime."""
    # 1. Official Synaptics Torq runtime package
    try:
        from torq.runtime import VMFBInferenceRunner
        return VMFBInferenceRunner
    except ImportError:
        pass

    try:
        from torq_runtime import VMFBInferenceRunner
        return VMFBInferenceRunner
    except ImportError:
        pass

    # 2. Check standard system candidate paths
    candidates = [
        "/usr/lib/python3.12/site-packages",
        "/usr/local/lib/python3.12/site-packages",
        "/usr/lib/python3/dist-packages",
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and candidate not in sys.path:
            sys.path.append(candidate)

    try:
        from torq.runtime import VMFBInferenceRunner
        return VMFBInferenceRunner
    except ImportError:
        pass

    # 3. Direct IREE Runtime package
    try:
        from iree.runtime import VMFBInferenceRunner
        return VMFBInferenceRunner
    except ImportError:
        pass

    return None


class TorqVisionEngine:
    """IREE Runtime Inference Engine for Synaptics Torq NPU."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.runner = None
        self._is_ready = False

    def load(self) -> None:
        """Initialize the IREE runtime session and load the compiled VMFB model."""
        # 1. If configured model path doesn't exist, attempt auto-discovery
        if not os.path.exists(self.model_path):
            model_dir = os.path.dirname(self.model_path) or "models"
            if os.path.isdir(model_dir):
                candidates = [
                    os.path.join(model_dir, f)
                    for f in os.listdir(model_dir)
                    if f.endswith((".vmfb", ".synap"))
                ]
                if candidates:
                    # Sort by modification time to pick the most recently downloaded model
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    self.model_path = candidates[0]
                    logger.info(f"Auto-discovered model artifact: {self.model_path}")

        logger.info(f"Loading Torq NPU model from: {self.model_path}")

        VMFBInferenceRunner = _import_vmfb_runner()
        if VMFBInferenceRunner is None:
            logger.error(
                "Torq runtime / IREE runtime not found in current Python environment! "
                "Ensure torq-runtime wheel is installed on the Coralboard SL2619."
            )
            return

        if not os.path.exists(self.model_path):
            logger.error(
                f"Model file not found at '{self.model_path}' and no .vmfb files found in 'models/'.\n"
                "Please run 'python3 download_model.py' to download your model from Frigate+."
            )
            return

        try:
            self.runner = VMFBInferenceRunner(self.model_path)
            self._is_ready = True
            logger.info(f"Synaptics Torq NPU session loaded successfully with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Torq NPU session for '{self.model_path}': {e}", exc_info=True)
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
        if not self._is_ready or self.runner is None:
            raise RuntimeError(
                f"NPU engine is not initialized. Model '{self.model_path}' is not loaded."
            )

        # Execute inference through the IREE VMFB runner
        outputs = self.runner.infer([input_tensor])
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        return [np.asarray(out) for out in outputs]
