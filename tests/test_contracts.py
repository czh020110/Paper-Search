from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from paper_search.config import _load_private_config, _resolve, load_settings
from paper_search.query_understanding import build_query_plan
from paper_search.retrieval.mock_backend import retrieve_mock_papers


class ContractTests(unittest.TestCase):
    def test_settings_snapshot_contains_runtime_and_experiment(self) -> None:
        settings = load_settings()
        snapshot = settings.config_snapshot()
        self.assertIn("runtime", snapshot)
        self.assertIn("experiment", snapshot)
        self.assertEqual(snapshot["runtime"]["log_level"], settings.log_level)

    def test_query_plan_recognizes_three_query_types(self) -> None:
        semantic = build_query_plan("2022年后关于大模型幻觉控制、使用强化学习方法、在CVPR发表的论文")
        metadata = build_query_plan("2023年发表于CVPR的论文，作者是Alice Zhang")
        navigational = build_query_plan("论文《SPAR: Multi-Agent Scientific Paper Retrieval》讲了什么？")

        self.assertEqual(semantic.intent_analysis.query_type, "semantic")
        self.assertEqual(metadata.intent_analysis.query_type, "metadata")
        self.assertEqual(navigational.intent_analysis.query_type, "navigational")

    def test_mock_backend_normalizes_and_deduplicates_candidates(self) -> None:
        fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
        query_plan = build_query_plan("2022年后关于大模型幻觉控制、使用强化学习方法、在CVPR发表的论文")
        papers, _ = retrieve_mock_papers(query_plan, fixtures_dir)
        ids = {paper.id for paper in papers}

        self.assertIn("semantic_scholar:rlhf-vlm-hallucination", ids)
        self.assertIn("openalex:rlhf-vlm-hallucination-duplicate", ids)
        self.assertTrue(any((paper.venue or "") == "CVPR" for paper in papers))


class TestConfigResolver(unittest.TestCase):
    def test_env_var_takes_precedence(self) -> None:
        result = _resolve("cache_dir", "CACHE_DIR", {"cache_dir": "/private/cache"}, "/default/cache")
        os.environ["CACHE_DIR"] = "/env/cache"
        try:
            env_result = _resolve("cache_dir", "CACHE_DIR", {"cache_dir": "/private/cache"}, "/default/cache")
            self.assertEqual(env_result, "/env/cache")
        finally:
            del os.environ["CACHE_DIR"]

    def test_private_config_over_default(self) -> None:
        # Ensure env var is not set
        os.environ.pop("OUTPUT_DIR", None)
        result = _resolve("output_dir", "OUTPUT_DIR", {"output_dir": "/private/output"}, "/default/output")
        self.assertEqual(result, "/private/output")

    def test_default_when_no_env_or_private(self) -> None:
        os.environ.pop("LOG_LEVEL", None)
        result = _resolve("log_level", "LOG_LEVEL", {}, "INFO")
        self.assertEqual(result, "INFO")

    def test_missing_private_config_file_harmless(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _load_private_config()
                self.assertEqual(result, {})
            finally:
                os.chdir(old_cwd)

    def test_config_snapshot_reflects_resolved_values(self) -> None:
        os.environ.pop("CACHE_DIR", None)
        os.environ.pop("OUTPUT_DIR", None)
        settings = load_settings()
        snapshot = settings.config_snapshot()
        # Defaults should appear
        self.assertIn(".cache", snapshot["runtime"]["cache_dir"])
        self.assertIn("outputs", snapshot["runtime"]["output_dir"])

    def test_private_config_loads_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.local.json"
            config_file.write_text(json.dumps({"log_level": "DEBUG", "output_dir": "/custom/output"}), encoding="utf-8")
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = _load_private_config()
                self.assertEqual(result.get("log_level"), "DEBUG")
                self.assertEqual(result.get("output_dir"), "/custom/output")
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
