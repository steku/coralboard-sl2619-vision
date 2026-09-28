#!/usr/bin/env python3
"""Download model artifacts from Frigate+ using an API key stored in config.yaml."""

import argparse
import json
import logging
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import zipfile
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("frigate_plus_downloader")

PLUS_API_HOST = "https://api.frigate.video"
API_KEY_REGEX = re.compile(
    r"^[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}:[a-z0-9]{40}$",
    re.IGNORECASE,
)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from config.yaml (or config.json), with environment variable fallback."""
    config: Dict[str, Any] = {}

    if os.path.exists(config_path):
        try:
            # Try PyYAML if installed
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except ImportError:
            # Fallback to simple parser or JSON if PyYAML is not yet installed
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                # Basic line-by-line parser for simple yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    current_section = None
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if ":" in line:
                            k, v = line.split(":", 1)
                            k, v = k.strip(), v.strip().strip("\"'")
                            if not v:
                                current_section = k
                                config.setdefault(current_section, {})
                            elif current_section:
                                config[current_section][k] = v
                            else:
                                config[k] = v
    elif os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

    # Allow environment variable override
    env_api_key = os.environ.get("PLUS_API_KEY")
    if env_api_key:
        config.setdefault("frigate_plus", {})["api_key"] = env_api_key

    return config


class FrigatePlusClient:
    """Client for authenticating and fetching models from Frigate+."""

    def __init__(self, api_key: str, host: str = PLUS_API_HOST):
        api_key = api_key.strip()
        if not API_KEY_REGEX.match(api_key):
            raise ValueError(
                "Invalid Frigate+ API key format. Expected format:\n"
                "  <uuid>:<40-char-secret>\n"
                "e.g. 12345678-1234-1234-1234-123456789abc:0123456789abcdef0123456789abcdef01234567"
            )
        self.api_key = api_key
        self.host = host.rstrip("/")
        self.access_token: Optional[str] = None

    def authenticate(self) -> None:
        """Authenticate with basic auth and retrieve JWT bearer token."""
        client_id, client_secret = self.api_key.split(":", 1)
        url = f"{self.host}/v1/auth/token"

        logger.info("Authenticating with Frigate+ API...")
        resp = requests.get(url, auth=(client_id, client_secret), timeout=15)
        if not resp.ok:
            raise RuntimeError(
                f"Failed to authenticate with Frigate+ (HTTP {resp.status_code}): {resp.text}"
            )

        data = resp.json()
        self.access_token = data.get("accessToken")
        if not self.access_token:
            raise RuntimeError("Authentication response did not contain an accessToken.")
        logger.info("Authentication successful.")

    def _headers(self) -> Dict[str, str]:
        if not self.access_token:
            self.authenticate()
        return {"Authorization": f"Bearer {self.access_token}"}

    def list_models(self) -> list:
        """Fetch all models including account models and Frigate+ base models."""
        models: list = []
        seen_ids = set()

        def _add_model(m: Dict[str, Any], is_base: bool = False) -> None:
            mid = m.get("id")
            if not mid or mid in seen_ids:
                return
            seen_ids.add(mid)
            if is_base or m.get("isBase") or m.get("base") or "base" in str(m.get("name", "")).lower():
                m["is_base"] = True
            else:
                m.setdefault("is_base", False)
            models.append(m)

        # 1. Primary account & global model list
        url = f"{self.host}/v1/model/list"
        resp = requests.get(url, headers=self._headers(), timeout=15)
        if resp.ok:
            data = resp.json()
            if isinstance(data, list):
                for m in data:
                    _add_model(m)
            elif isinstance(data, dict):
                # Standard account models
                for m in data.get("list") or []:
                    _add_model(m)
                # Check for base models attached to response
                for key in ("base", "base_models", "baseModels", "public_models"):
                    for m in data.get(key) or []:
                        _add_model(m, is_base=True)
        else:
            logger.warning(f"Failed to fetch model/list: {resp.status_code}")

        # 2. Query explicit base model endpoints if available
        for endpoint in ("model/base", "model/base_models", "model/list?include_base=true"):
            try:
                b_resp = requests.get(f"{self.host}/v1/{endpoint}", headers=self._headers(), timeout=10)
                if b_resp.ok:
                    b_data = b_resp.json()
                    items = b_data if isinstance(b_data, list) else (
                        b_data.get("list") or b_data.get("models") or b_data.get("base") or []
                    )
                    for m in items:
                        _add_model(m, is_base=True)
            except Exception:
                pass

        return models

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Fetch metadata for a specific model."""
        url = f"{self.host}/v1/model/{model_id}"
        resp = requests.get(url, headers=self._headers(), timeout=15)
        if not resp.ok:
            raise RuntimeError(f"Failed to get model info for {model_id}: {resp.text}")
        return resp.json()

    def get_download_url(self, model_id: str) -> str:
        """Request presigned download URL for the model binary."""
        url = f"{self.host}/v1/model/{model_id}/signed_url"
        resp = requests.get(url, headers=self._headers(), timeout=15)
        if not resp.ok:
            raise RuntimeError(f"Failed to get signed URL for {model_id}: {resp.text}")
        data = resp.json()
        download_url = data.get("url")
        if not download_url:
            raise RuntimeError(f"Signed URL payload missing 'url' key: {data}")
        return str(download_url)


def download_file(url: str, dest_path: Path) -> None:
    """Stream download a URL directly to the destination path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")

    logger.info(f"Downloading artifact to {dest_path}...")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(
                            f"\rProgress: {downloaded / (1024*1024):.2f}MB / "
                            f"{total_size / (1024*1024):.2f}MB ({percent:.1f}%)",
                            end="",
                            flush=True,
                        )

    print()
    tmp_path.replace(dest_path)
    logger.info(f"Model saved successfully to: {dest_path}")


def is_coralboard_compatible(model: Dict[str, Any]) -> bool:
    """Check if model is compatible with Coralboard SL2619 Torq NPU or is a Frigate+ base model."""
    # 1. Base models are pre-trained foundational models available to all subscribers
    if model.get("is_base") or model.get("base") or model.get("isBase"):
        return True
    name = str(model.get("name", "")).lower()
    if "base" in name:
        return True

    # 2. Check supportedDetectors metadata
    supported = [str(d).lower() for d in (model.get("supportedDetectors") or [])]
    if not supported:
        # If no specific detector restriction is tagged, allow it
        return True

    # 3. Matches Synaptics Torq NPU directly
    if any(d in supported for d in ("synaptics", "torq")):
        return True

    # 4. OpenVINO, ONNX, CPU base formats that run via IREE / Torq
    if any(d in supported for d in ("onnx", "openvino", "cpu")):
        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model from Frigate+ using config.yaml API key.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Specific model ID to download (overrides config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for model file (default: models)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models (including base models) and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if the model is not flagged as compatible",
    )
    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.config)
    plus_cfg = config.get("frigate_plus", {})

    api_key = plus_cfg.get("api_key", "").strip()
    if not api_key or "00000000-0000" in api_key:
        logger.error(
            f"No valid API key found in '{args.config}'.\n"
            "Please copy config.example.yaml to config.yaml and configure your real Frigate+ API key:\n"
            "  cp config.example.yaml config.yaml"
        )
        sys.exit(1)

    model_id = args.model_id or plus_cfg.get("model_id", "").strip()
    output_dir = Path(args.output_dir or plus_cfg.get("output_dir", "models"))

    # 2. Initialize client
    client = FrigatePlusClient(api_key=api_key)
    client.authenticate()

    # 3. List models
    models = client.list_models()
    if not models:
        logger.warning("No models found in Frigate+.")
        return

    if args.list:
        print("\n--- Frigate+ Models (Account & Base Models) ---")
        for m in models:
            mid = m.get("id")
            name = m.get("name", "unnamed")
            status = m.get("status", "unknown")
            supported = m.get("supportedDetectors") or []
            is_base = m.get("is_base", False)
            source_tag = "Base Model" if is_base else "Account Model"
            compat = "✓ Compatible" if is_coralboard_compatible(m) else f"✗ Incompatible (Targets: {', '.join(supported) if supported else 'unspecified'})"
            targets_str = f"Targets: {', '.join(supported)}" if supported else "Universal/Base"
            print(f"- ID: {mid:<36} | {name:<22} | [{source_tag:<13}] | [{targets_str:<25}] | [{compat}]")
        return

    # Filter for compatible models (including base models)
    compatible_models = [m for m in models if is_coralboard_compatible(m)]

    # Select target model
    target_model_id = model_id
    if not target_model_id:
        if not compatible_models:
            logger.error(
                "No compatible models found in Frigate+.\n"
                f"Found {len(models)} model(s): {[m.get('name') for m in models]}.\n"
                "Use --list to inspect models, or use --model-id <id> --force to download anyway."
            )
            sys.exit(1)

        # Prioritize:
        # 1. Custom account model compiled for synaptics/torq
        # 2. Pre-trained base models
        # 3. Any compatible model
        account_synaptics = [
            m for m in compatible_models
            if not m.get("is_base") and any(d in [str(x).lower() for x in (m.get("supportedDetectors") or [])] for d in ("synaptics", "torq"))
        ]
        if account_synaptics:
            chosen = account_synaptics[0]
            logger.info(f"Selected latest account Synaptics NPU model: {chosen.get('id')} ({chosen.get('name')})")
        else:
            chosen = compatible_models[0]
            m_type = "Base Model" if chosen.get("is_base") else "Model"
            logger.info(f"Selected {m_type}: {chosen.get('id')} ({chosen.get('name')})")

        target_model_id = chosen.get("id")
    else:
        # User specified explicit model ID, verify compatibility
        matched = next((m for m in models if m.get("id") == target_model_id), None)
        if matched and not is_coralboard_compatible(matched) and not args.force:
            logger.warning(
                f"Model '{target_model_id}' does not list compatible detectors "
                f"(supported: {matched.get('supportedDetectors')}).\n"
                "Pass --force to proceed anyway."
            )
            sys.exit(1)

    # 4. Fetch model metadata
    model_info = client.get_model_info(target_model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata JSON
    info_path = output_dir / f"{target_model_id}.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Saved model metadata to {info_path}")

    # 5. Fetch signed download URL
    download_url = client.get_download_url(target_model_id)

    # Determine file extension/name from presigned URL path or fallback to .vmfb
    url_path = download_url.split("?")[0]
    url_filename = os.path.basename(url_path)
    if url_filename and "." in url_filename:
        file_name = url_filename
    else:
        file_name = f"{target_model_id}.vmfb"

    dest_path = output_dir / file_name

    # 6. Stream download
    download_file(download_url, dest_path)

    # 7. Inspect downloaded artifact and unpack if archive
    final_model_path = dest_path
    if dest_path.exists():
        with open(dest_path, "rb") as f:
            header = f.read(128)
        if header.startswith(b"<?xml") or b"<Error>" in header:
            err_snippet = header.decode("utf-8", errors="ignore")
            logger.error(f"Downloaded artifact is an XML error document from storage:\n{err_snippet}")
            sys.exit(1)
        if header.startswith(b"{") and (b"error" in header.lower() or b"message" in header.lower()):
            err_snippet = header.decode("utf-8", errors="ignore")
            logger.error(f"Downloaded artifact contains JSON error response:\n{err_snippet}")
            sys.exit(1)

        if header.startswith(b"\x1f\x8b") or tarfile.is_tarfile(dest_path):
            logger.info("Artifact is a tar archive. Extracting...")
            try:
                with tarfile.open(dest_path, "r:*") as tar:
                    tar.extractall(path=output_dir)
                logger.info("Tar archive extracted successfully.")
            except Exception as e:
                logger.warning(f"Failed to extract tar archive: {e}")
        elif header.startswith(b"PK\x03\x04") or zipfile.is_zipfile(dest_path):
            logger.info("Artifact is a zip archive. Extracting...")
            try:
                with zipfile.ZipFile(dest_path, "r") as zf:
                    zf.extractall(path=output_dir)
                logger.info("Zip archive extracted successfully.")
            except Exception as e:
                logger.warning(f"Failed to extract zip archive: {e}")

        # Look for extracted model file (.vmfb or .synap)
        for ext in (".vmfb", ".synap"):
            for candidate in output_dir.glob(f"*{ext}"):
                if candidate.name not in ("yolov9_320.vmfb", "model.vmfb") and candidate.is_file():
                    final_model_path = candidate
                    break

    # 8. Create convenient aliases (models/yolov9_320.vmfb and models/model.vmfb)
    for alias_name in ("yolov9_320.vmfb", "model.vmfb"):
        alias_path = output_dir / alias_name
        if alias_path.resolve() != final_model_path.resolve():
            try:
                if alias_path.is_symlink() or alias_path.exists():
                    alias_path.unlink()
                alias_path.symlink_to(final_model_path.name)
                logger.info(f"Linked {alias_name} -> {final_model_path.name}")
            except Exception:
                pass

    logger.info("Done!")


if __name__ == "__main__":
    main()
