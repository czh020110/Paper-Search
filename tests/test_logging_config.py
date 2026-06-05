from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

from paper_search.logging_config import JsonFormatter, setup_logging


class TestJsonFormatter(unittest.TestCase):
    def test_output_is_valid_json(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="paper_search.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        self.assertIn("timestamp", data)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "paper_search.test")
        self.assertEqual(data["message"], "hello")

    def test_run_id_in_output(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="paper_search.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.run_id = "run_20260605_abc12345"  # type: ignore[attr-defined]
        output = formatter.format(record)
        data = json.loads(output)
        self.assertEqual(data["run_id"], "run_20260605_abc12345")

    def test_extra_fields_in_output(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="paper_search.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.run_id = "r1"  # type: ignore[attr-defined]
        record.stage = "initial_retrieval"  # type: ignore[attr-defined]
        record.api_source = "semantic_scholar"  # type: ignore[attr-defined]
        record.cache_hit = True  # type: ignore[attr-defined]
        output = formatter.format(record)
        data = json.loads(output)
        self.assertEqual(data["stage"], "initial_retrieval")
        self.assertEqual(data["api_source"], "semantic_scholar")
        self.assertTrue(data["cache_hit"])

    def test_exception_in_output(self) -> None:
        formatter = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord(
                name="paper_search.test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="error",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        self.assertIn("exception", data)
        self.assertIn("ValueError: boom", data["exception"])

    def test_missing_run_id_defaults_empty(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="paper_search.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        self.assertEqual(data["run_id"], "")


class TestSetupLogging(unittest.TestCase):
    def test_creates_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            setup_logging("run_test123", log_dir, "INFO")

            logger = logging.getLogger("paper_search")
            logger.info("test message", extra={"run_id": "run_test123"})
            # Flush all handlers to ensure the log file is written
            for handler in logger.handlers:
                handler.flush()

            log_file = log_dir / "pipeline.jsonl"
            self.assertTrue(log_file.exists())

            content = log_file.read_text(encoding="utf-8").strip()
            self.assertTrue(content)
            data = json.loads(content)
            self.assertEqual(data["message"], "test message")
            self.assertEqual(data["run_id"], "run_test123")

    def test_no_duplicate_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            setup_logging("run_test1", log_dir, "INFO")
            handler_count = len(logging.getLogger("paper_search").handlers)
            setup_logging("run_test2", log_dir, "INFO")
            self.assertEqual(len(logging.getLogger("paper_search").handlers), handler_count)

    def tearDown(self) -> None:
        # Clean up all handlers added during tests to avoid leakage across tests
        pkg_logger = logging.getLogger("paper_search")
        for h in list(pkg_logger.handlers):
            if getattr(h, "_paper_search_json", False) or getattr(h, "_paper_search_console", False):
                h.close()
                pkg_logger.removeHandler(h)


if __name__ == "__main__":
    unittest.main()
