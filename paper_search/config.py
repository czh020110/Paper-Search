from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget import BudgetLimits

PRIVATE_CONFIG_FILENAME = "config.local.json"


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    output_dir: Path
    log_level: str
    llm_api_base: str
    llm_model: str
    llm_fast_model: str
    runtime_config: dict[str, Any]
    experiment_config: dict[str, Any]

    def budget_limits(self) -> BudgetLimits:
        budget = self.experiment_config.get("selection", {}).get("budget", {})
        return BudgetLimits(
            max_api_calls=budget.get("max_api_calls"),
            max_tokens=budget.get("max_tokens"),
            max_rounds=budget.get("max_rounds"),
            max_pool_size=budget.get("max_pool_size"),
            max_batch_judgments=budget.get("max_batch_judgments"),
        )

    def config_snapshot(self) -> dict[str, Any]:
        return {
            "runtime": {
                "cache_dir": str(self.cache_dir),
                "output_dir": str(self.output_dir),
                "log_level": self.log_level,
                "llm_api_base": self.llm_api_base,
                "llm_model": self.llm_model,
                "llm_fast_model": self.llm_fast_model,
            },
            "experiment": self.experiment_config,
        }


def _load_private_config() -> dict[str, Any]:
    """Load a local private config file (``config.local.json``) from CWD.

    Returns an empty dict if the file does not exist.  The file is
    intended to be listed in ``.gitignore`` so secrets and local
    overrides never enter version control.
    """
    path = Path.cwd() / PRIVATE_CONFIG_FILENAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve(key: str, env_var: str, private_config: dict[str, Any], default: Any) -> Any:
    """Resolve a config value with 3-tier priority: env > private config > default."""
    env_value = os.getenv(env_var)
    if env_value is not None:
        return env_value
    if key in private_config:
        return private_config[key]
    return default


def load_settings() -> Settings:
    private = _load_private_config()

    cache_dir = Path(_resolve("cache_dir", "CACHE_DIR", private, ".cache"))
    output_dir = Path(_resolve("output_dir", "OUTPUT_DIR", private, "outputs"))
    log_level = _resolve("log_level", "LOG_LEVEL", private, "INFO")
    llm_api_base = _resolve("llm_api_base", "LLM_API_BASE", private, "https://api.openai.com/v1")
    llm_model = _resolve("llm_model", "LLM_MODEL", private, "gpt-4o-mini")
    llm_fast_model = _resolve("llm_fast_model", "LLM_FAST_MODEL", private, llm_model)

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
            "budget": {
                "max_api_calls": None,
                "max_tokens": None,
                "max_rounds": 3,
                "max_pool_size": 200,
                "max_batch_judgments": None,
            },
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
        llm_model=llm_model,
        llm_fast_model=llm_fast_model,
        runtime_config=runtime_config,
        experiment_config=experiment_config,
    )
