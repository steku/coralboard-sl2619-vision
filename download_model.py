#!/usr/bin/env python3
"""Download model artifacts from Frigate+ using an API key stored in config.yaml."""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
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
        """Fetch list of all trained models associated with the account."""
        url = f"{self.host}/v1/model/list"
        resp = requests.get(url, headers=self._headers(), timeout=15)
        if not resp.ok:
            raise RuntimeError(f"Failed to list models (HTTP {resp.status_code}): {resp.text}")
        data = resp.json()
        return data.get("list") or []

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
    """Check if the model metadata indicates support for Synaptics / Torq NPU."""
    supported = [str(d).lower() for d in (model.get("supportedDetectors") or [])]
    return any(d in supported for d in ("synaptics", "torq"))


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
        help="List available models in account with compatibility status and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if the model is not flagged as Coralboard SL2619 compatible",
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
        logger.warning("No models found in your Frigate+ account.")
        return

    if args.list:
        print("\n--- Frigate+ Models (Coralboard SL2619 NPU Compatibility) ---")
        for m in models:
            mid = m.get("id")
            name = m.get("name", "unnamed")
            status = m.get("status", "unknown")
            created = m.get("createdAt", "")
            supported = m.get("supportedDetectors") or []
            compat = "✓ Compatible (Synaptics NPU)" if is_coralboard_compatible(m) else f"✗ Incompatible (Targets: {', '.join(supported) if supported else 'unspecified'})"
            print(f"- ID: {mid:<36} | {name:<20} | Status: {status:<10} | [{compat}]")
        return

    # Filter for Coralboard SL2619 compatible models
    compatible_models = [m for m in models if is_coralboard_compatible(m)]

    # Select target model
    target_model_id = model_id
    if not target_model_id:
        if not compatible_models:
            logger.error(
                "No Coralboard SL2619 compatible models ('synaptics' detector) found in your Frigate+ account.\n"
                f"Found {len(models)} model(s) targeting other hardware: "
                f"{[m.get('supportedDetectors') for m in models]}.\n"
                "Use --list to inspect models, or use --model-id <id> --force to download anyway."
            )
            sys.exit(1)

        target_model_id = compatible_models[0].get("id")
        logger.info(f"Selected latest compatible Synaptics NPU model: {target_model_id}")
    else:
        # User specified explicit model ID, verify compatibility
        matched = next((m for m in models if m.get("id") == target_model_id), None)
        if matched and not is_coralboard_compatible(matched) and not args.force:
            logger.warning(
                f"Model '{target_model_id}' does not list 'synaptics' in supportedDetectors "
                f"(supported: {matched.get('supportedDetectors')}).\n"
                "It may not execute on the Coralboard SL2619 Torq NPU. Pass --force to proceed anyway."
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
    logger.info("Done!")


if __name__ == "__main__":
    main()
