from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_search.config import load_settings
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


if __name__ == "__main__":
    unittest.main()
