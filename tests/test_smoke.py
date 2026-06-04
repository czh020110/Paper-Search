from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_search.config import load_settings
from paper_search.pipeline import run_pipeline


SEMANTIC_QUERY = "2022年后关于大模型幻觉控制、使用强化学习方法、在CVPR发表的论文"
NAV_QUERY = "论文《SPAR: Multi-Agent Scientific Paper Retrieval》讲了什么？"


class SmokeTests(unittest.TestCase):
    def test_pipeline_outputs_expected_artifacts(self) -> None:
        settings = load_settings()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_pipeline(SEMANTIC_QUERY, "mock", Path(tmp_dir), settings)
            output_files = result["output_files"]

            self.assertTrue(Path(output_files["markdown"]).exists())
            self.assertTrue(Path(output_files["graph"]).exists())
            self.assertTrue(Path(output_files["experiment"]).exists())

            experiment = json.loads(Path(output_files["experiment"]).read_text(encoding="utf-8"))
            self.assertIn("config_snapshot", experiment)
            self.assertIn("stage_metrics", experiment)
            self.assertIn("overall", experiment["stage_metrics"])
            self.assertIn("initial_retrieval", experiment["stage_metrics"])
            self.assertEqual(experiment["stage_metrics"]["overall"]["f1"], 1.0)
            self.assertGreaterEqual(len(result["papers"]), 3)
            self.assertGreaterEqual(result["evaluation"]["recall"], 1.0)

    def test_pipeline_deduplicates_same_doi(self) -> None:
        settings = load_settings()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_pipeline(SEMANTIC_QUERY, "mock", Path(tmp_dir), settings)
            paper_ids = [paper["id"] for paper in result["papers"]]
            self.assertIn("semantic_scholar:rlhf-vlm-hallucination", paper_ids)
            self.assertNotIn("openalex:rlhf-vlm-hallucination-duplicate", paper_ids)

    def test_navigational_query_hits_exact_title(self) -> None:
        settings = load_settings()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_pipeline(NAV_QUERY, "mock", Path(tmp_dir), settings)
            paper_ids = [paper["id"] for paper in result["papers"]]
            self.assertIn("semantic_scholar:spar-paper", paper_ids)
            self.assertEqual(result["query_plan"]["intent_analysis"]["query_type"], "navigational")

    def test_graph_and_markdown_follow_contract(self) -> None:
        settings = load_settings()
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_pipeline(SEMANTIC_QUERY, "mock", Path(tmp_dir), settings)
            graph = json.loads(Path(result["output_files"]["graph"]).read_text(encoding="utf-8"))
            markdown = Path(result["output_files"]["markdown"]).read_text(encoding="utf-8")

            self.assertEqual(graph["query"], SEMANTIC_QUERY)
            self.assertIn("nodes", graph)
            self.assertIn("edges", graph)
            self.assertTrue(all("relevance" in node for node in graph["nodes"]))
            self.assertIn("# 查询摘要", markdown)
            self.assertIn("# 高度相关论文列表", markdown)
            self.assertIn("# 部分相关论文列表", markdown)
            self.assertIn("# 引文关系说明", markdown)
            self.assertIn("# 运行摘要", markdown)

    def test_golden_set_uses_two_level_relevance_labels(self) -> None:
        golden_set_path = Path(__file__).resolve().parent.parent / "fixtures" / "golden_set.json"
        payload = json.loads(golden_set_path.read_text(encoding="utf-8"))
        cases = payload["cases"]

        self.assertGreaterEqual(len(cases), 12)
        for case in cases:
            self.assertIn("highly_relevant_papers", case)
            self.assertIn("partially_relevant_papers", case)


if __name__ == "__main__":
    unittest.main()
