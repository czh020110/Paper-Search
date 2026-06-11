"""AstaBench 数据集下载与缓存。"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_ASTA_REPO = "allenai/asta-bench"
_ASTA_REVISION = "main"
_PF_DATA_FILE = "tasks/paper_finder_bench/validation_2025_05.json"
_NORMALIZER_FILE = "tasks/paper_finder_bench/normalizer_reference.json"


def _cache_dir() -> Path:
    d = Path(os.getenv("CACHE_DIR", ".cache")) / "asta-bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_dataset(token: str | None = None) -> Path:
    """Download paper_finder_bench validation data, return local path."""
    from huggingface_hub import hf_hub_download

    local = _cache_dir() / "validation_2025_05.json"
    if local.exists():
        logger.info("Dataset already cached at %s", local)
        return local

    if not token:
        token = os.getenv("HF_TOKEN")

    logger.info("Downloading AstaBench paper_finder_bench validation data...")
    path = hf_hub_download(
        repo_id=_ASTA_REPO,
        filename=_PF_DATA_FILE,
        repo_type="dataset",
        token=token,
        revision=_ASTA_REVISION,
    )
    # Copy to cache
    local.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("Dataset cached at %s", local)
    return local


def download_normalizer(token: str | None = None) -> dict[str, int]:
    """Download normalizer_reference.json (total relevant papers per query)."""
    from huggingface_hub import hf_hub_download

    local = _cache_dir() / "normalizer_reference.json"
    if not local.exists():
        if not token:
            token = os.getenv("HF_TOKEN")
        try:
            path = hf_hub_download(
                repo_id=_ASTA_REPO,
                filename=_NORMALIZER_FILE,
                repo_type="dataset",
                token=token,
                revision=_ASTA_REVISION,
            )
            local.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to download normalizer: %s", e)
            return {}

    try:
        return json.loads(local.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_dataset(token: str | None = None) -> list[dict]:
    """Load paper_finder_bench validation data. Download if not cached."""
    local = _cache_dir() / "validation_2025_05.json"
    if not local.exists():
        local = download_dataset(token)
    return json.loads(local.read_text(encoding="utf-8"))


def dataset_info() -> dict:
    """Return dataset status and stats."""
    local = _cache_dir() / "validation_2025_05.json"
    if not local.exists():
        return {"downloaded": False}

    data = json.loads(local.read_text(encoding="utf-8"))
    from collections import Counter
    types = Counter(s["input"]["query_id"].split("_")[0] for s in data)
    return {
        "downloaded": True,
        "total": len(data),
        "types": dict(types),
        "path": str(local),
    }
