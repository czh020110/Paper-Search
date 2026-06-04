from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    output_dir: Path
    log_level: str
    llm_api_base: str
    runtime_config: dict[str, Any]
    experiment_config: dict[str, Any]

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "runtime": {
                "cache_dir": str(self.cache_dir),
                "output_dir": str(self.output_dir),
                "log_level": self.log_level,
                "llm_api_base": self.llm_api_base,
            },
            "experiment": self.experiment_config,
        }


def load_settings() -> Settings:
    cache_dir = Path(os.getenv("CACHE_DIR", ".cache"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    log_level = os.getenv("LOG_LEVEL", "INFO")
    llm_api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")

    experiment_config = {
        "query_understanding": {
            "strategy": "rule-based-bootstrap",
            "query_types": ["navigational", "semantic", "metadata"],
        },
        "retrieval": {
            "backend": "mock",
            "target_seed_pool_size": [30, 50],
        },
        "selection": {
            "top_k": None,
            "threshold": None,
            "batch_size": None,
            "budget": None,
        },
        "evaluation": {
            "included_relevance_levels": ["highly_relevant_papers", "partially_relevant_papers"],
            "golden_set_target_size": [12, 18],
        },
    }

    runtime_config = {
        "semantic_scholar_api_key_set": bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
        "openalex_api_key_set": bool(os.getenv("OPENALEX_API_KEY")),
        "llm_api_key_set": bool(os.getenv("LLM_API_KEY")),
    }

    return Settings(
        cache_dir=cache_dir,
        output_dir=output_dir,
        log_level=log_level,
        llm_api_base=llm_api_base,
        runtime_config=runtime_config,
        experiment_config=experiment_config,
    )
