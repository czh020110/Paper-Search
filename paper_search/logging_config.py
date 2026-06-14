from __future__ import annotations

import json
import logging
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects for structured logging."""

    # Extra attributes that may be attached via logger.info(..., extra={...})
    _KNOWN_EXTRAS = ("stage", "domain", "paper_id", "api_source", "cache_hit", "budget_reason")

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, object] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", ""),
        }
        for key in self._KNOWN_EXTRAS:
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(run_id: str, log_dir: Path, log_level: str = "INFO") -> None:
    """Configure structured JSON logging for a pipeline run.

    Adds a file handler writing ``pipeline.jsonl`` under *log_dir* with
    :class:`JsonFormatter`, and a console handler (stderr) for developer visibility.
    """
    # run_id is injected into log records via the JsonFormatter above
    _ = run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    pkg_logger = logging.getLogger("paper_search")
    pkg_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Close any old JSON file handler from previous runs (always create a fresh one)
    for h in list(pkg_logger.handlers):
        if getattr(h, "_paper_search_json", False):
            h.close()
            pkg_logger.removeHandler(h)

    file_handler = logging.FileHandler(log_dir / "pipeline.jsonl", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    file_handler._paper_search_json = True  # type: ignore[attr-defined]
    pkg_logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_paper_search_console", False) for h in pkg_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        console_handler._paper_search_console = True  # type: ignore[attr-defined]
        pkg_logger.addHandler(console_handler)
