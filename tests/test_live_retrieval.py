"""Tests for live retrieval backends (Semantic Scholar & OpenAlex)."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from paper_search.contracts import IntentAnalysis, Paper, QueryPlan
from paper_search.retrieval.live_backend import retrieve_live_papers
from paper_search.retrieval.openalex import _paper_from_oa, _reconstruct_abstract, search_works
from paper_search.retrieval.semantic_scholar import _paper_from_s2, search_papers


S2_SEARCH_RESPONSE: dict[str, Any] = {
    "total": 2,
    "offset": 0,
    "data": [
        {
            "paperId": "abc123",
            "title": "RLHF: Learning from Human Feedback",
            "abstract": "We present a method for learning from human feedback.",
            "authors": [
                {"authorId": "1", "name": "Alice Smith"},
                {"authorId": "2", "name": "Bob Jones"},
            ],
            "year": 2023,
            "venue": "NeurIPS",
            "publicationDate": "2023-12-01",
            "url": "https://semanticscholar.org/paper/abc123",
            "openAccessPdf": {"url": "https://arxiv.org/pdf/2301.00001", "status": "GREEN"},
            "fieldsOfStudy": ["Computer Science", "AI"],
            "citationCount": 42,
            "referenceCount": 30,
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2301.00001", "OpenAlex": "W123"},
        },
        {
            "paperId": "def456",
            "title": "Hallucination Control in Large Models",
            "abstract": None,
            "authors": [],
            "year": 2022,
            "venue": "CVPR",
            "publicationDate": "2022-06-15",
            "url": "https://semanticscholar.org/paper/def456",
            "openAccessPdf": None,
            "fieldsOfStudy": None,
            "citationCount": 10,
            "referenceCount": 5,
            "externalIds": {"DOI": None, "ArXiv": None},
        },
    ],
}

OA_SEARCH_RESPONSE: dict[str, Any] = {
    "meta": {"count": 2, "cursor": "next_cursor_value"},
    "results": [
        {
            "id": "https://openalex.org/W999",
            "ids": {"openalex": "https://openalex.org/W999", "doi": "https://doi.org/10.5678/oa-test", "arxiv": None},
            "title": "Vision Transformers for Image Recognition",
            "display_name": "Vision Transformers for Image Recognition",
            "publication_year": 2024,
            "publication_date": "2024-03-15",
            "cited_by_count": 100,
            "referenced_works_count": 50,
            "authorships": [
                {"author": {"id": "https://openalex.org/A1", "display_name": "Carol White"}},
            ],
            "primary_location": {
                "source": {"id": "https://openalex.org/P1", "display_name": "CVPR"},
                "landing_page_url": None,
                "pdf_url": None,
            },
            "open_access": {"oa_url": "https://arxiv.org/pdf/2403.00001", "oa_status": "green"},
            "oa_locations": [{"pdf_url": "https://arxiv.org/pdf/2403.00001", "oa_status": "green"}],
            "topics": [{"id": "T1", "display_name": "Computer Vision", "score": 0.9}],
            "concepts": [{"id": "C1", "display_name": "Machine Learning", "score": 0.8}],
            "abstract_inverted_index": {
                "We": [0], "present": [1], "ViT": [2], "for": [3], "images": [4],
            },
        },
        {
            "id": "https://openalex.org/W888",
            "ids": {"openalex": "https://openalex.org/W888", "doi": None, "arxiv": "2302.00002"},
            "title": "Reinforcement Learning for NLP",
            "display_name": "Reinforcement Learning for NLP",
            "publication_year": 2023,
            "publication_date": "2023-07-01",
            "cited_by_count": 55,
            "referenced_works_count": 20,
            "authorships": [],
            "primary_location": None,
            "open_access": None,
            "oa_locations": [],
            "topics": [],
            "concepts": [],
            "abstract_inverted_index": None,
        },
    ],
}


def _make_query_plan(query_type: str = "semantic") -> QueryPlan:
    return QueryPlan(
        original_query="2022年后关于大模型幻觉控制的论文",
        intent_analysis=IntentAnalysis(
            domain="Academic Search",
            query_type=query_type,
            boundary_note="检索阶段先按 >=2022 处理",
        ),
        hard_filters={
            "year": {
                "operator": ">=",
                "value": 2022,
                "relaxed_window": [2022, 2026],
            }
        },
        ranking_signals={
            "preferred_venues": ["CVPR", "NeurIPS"],
            "venue_match_mode": "fuzzy_match_and_bonus",
            "venue_as_hard_filter": False,
        },
        semantic_queries={
            "core_concepts": ["hallucination", "large language model", "control"],
            "methodologies": ["reinforcement learning"],
        },
        sub_queries_for_retrieval=[
            "hallucination control large language model",
            "LLM hallucination mitigation",
        ],
        api_payload_translation={
            "semantic_scholar": [
                {"query": "hallucination control large language model", "year": "2022-"},
                {"query": "LLM hallucination mitigation", "year": "2022-"},
            ],
            "openalex": [
                {"search": "hallucination control large language model", "filter": "publication_year:>2022"},
                {"search": "LLM hallucination mitigation", "filter": "publication_year:>2022"},
            ],
        },
    )


class TestSemanticScholarParser(unittest.TestCase):
    def test_paper_from_s2_full(self) -> None:
        raw = S2_SEARCH_RESPONSE["data"][0]
        paper = _paper_from_s2(raw)

        self.assertEqual(paper.id, "semantic_scholar:abc123")
        self.assertEqual(paper.title, "RLHF: Learning from Human Feedback")
        self.assertEqual(paper.abstract, "We present a method for learning from human feedback.")
        self.assertEqual(len(paper.authors), 2)
        self.assertEqual(paper.authors[0]["name"], "Alice Smith")
        self.assertEqual(paper.year, 2023)
        self.assertEqual(paper.venue, "NeurIPS")
        self.assertEqual(paper.citation_count, 42)
        self.assertEqual(paper.source_api, "semantic_scholar")
        self.assertEqual(paper.source_ids["doi"], "10.1234/test")
        self.assertEqual(paper.source_ids["arxiv"], "2301.00001")
        self.assertIsNotNone(paper.open_access_pdf)
        self.assertEqual(paper.open_access_pdf["url"], "https://arxiv.org/pdf/2301.00001")
        self.assertEqual(paper.fields, ["Computer Science", "AI"])

    def test_paper_from_s2_minimal(self) -> None:
        raw = S2_SEARCH_RESPONSE["data"][1]
        paper = _paper_from_s2(raw)

        self.assertEqual(paper.id, "semantic_scholar:def456")
        self.assertEqual(paper.title, "Hallucination Control in Large Models")
        self.assertIsNone(paper.abstract)
        self.assertEqual(paper.authors, [])
        self.assertEqual(paper.venue, "CVPR")
        self.assertIsNone(paper.open_access_pdf)
        self.assertEqual(paper.fields, [])

    @patch("paper_search.retrieval.semantic_scholar.httpx.get")
    def test_search_parsers(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = S2_SEARCH_RESPONSE
        mock_get.return_value = mock_response

        papers = search_papers(query="reinforcement learning", year_from=2022, limit=20)

        self.assertEqual(len(papers), 2)
        self.assertIsInstance(papers[0], Paper)
        mock_get.assert_called_once()


class TestOpenAlexParser(unittest.TestCase):
    def test_paper_from_oa_full(self) -> None:
        raw = OA_SEARCH_RESPONSE["results"][0]
        paper = _paper_from_oa(raw)

        self.assertEqual(paper.id, "openalex:https://openalex.org/W999")
        self.assertEqual(paper.title, "Vision Transformers for Image Recognition")
        self.assertEqual(paper.abstract, "We present ViT for images")
        self.assertEqual(paper.year, 2024)
        self.assertEqual(paper.venue, "CVPR")
        self.assertEqual(paper.citation_count, 100)
        self.assertEqual(paper.source_api, "openalex")
        self.assertEqual(paper.source_ids["doi"], "10.5678/oa-test")
        self.assertIsNotNone(paper.open_access_pdf)
        self.assertEqual(paper.topics, ["Computer Vision"])
        self.assertEqual(paper.fields, ["Machine Learning"])

    def test_paper_from_oa_minimal(self) -> None:
        raw = OA_SEARCH_RESPONSE["results"][1]
        paper = _paper_from_oa(raw)

        self.assertEqual(paper.id, "openalex:https://openalex.org/W888")
        self.assertEqual(paper.title, "Reinforcement Learning for NLP")
        self.assertIsNone(paper.abstract)
        self.assertIsNone(paper.venue)
        self.assertEqual(paper.authors, [])
        self.assertEqual(paper.topics, [])
        self.assertEqual(paper.fields, [])

    def test_reconstruct_abstract(self) -> None:
        inverted = {"We": [0], "present": [1], "a": [2], "method": [3]}
        result = _reconstruct_abstract(inverted)
        self.assertEqual(result, "We present a method")

    def test_reconstruct_abstract_empty(self) -> None:
        self.assertIsNone(_reconstruct_abstract(None))
        self.assertIsNone(_reconstruct_abstract({}))

    @patch("paper_search.retrieval.openalex.httpx.get")
    def test_search_works(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = OA_SEARCH_RESPONSE
        mock_get.return_value = mock_response

        papers = search_works(query="vision transformer", year_from=2023, per_page=25)

        self.assertEqual(len(papers), 2)
        self.assertIsInstance(papers[0], Paper)
        mock_get.assert_called_once()


class TestLiveBackendIntegration(unittest.TestCase):
    @patch("paper_search.retrieval.semantic_scholar.httpx.get")
    @patch("paper_search.retrieval.openalex.httpx.get")
    def test_retrieve_live_papers_semantic(self, mock_oa_get: MagicMock, mock_s2_get: MagicMock) -> None:
        s2_response = MagicMock()
        s2_response.status_code = 200
        s2_response.json.return_value = S2_SEARCH_RESPONSE
        mock_s2_get.return_value = s2_response

        oa_response = MagicMock()
        oa_response.status_code = 200
        oa_response.json.return_value = OA_SEARCH_RESPONSE
        mock_oa_get.return_value = oa_response

        query_plan = _make_query_plan("semantic")
        papers, edges = retrieve_live_papers(query_plan)

        self.assertGreater(len(papers), 0)
        self.assertIsInstance(papers[0], Paper)

    @patch("paper_search.retrieval.semantic_scholar.httpx.get")
    @patch("paper_search.retrieval.openalex.httpx.get")
    def test_retrieve_live_papers_navigational(self, mock_oa_get: MagicMock, mock_s2_get: MagicMock) -> None:
        s2_response = MagicMock()
        s2_response.status_code = 200
        s2_response.json.return_value = S2_SEARCH_RESPONSE
        mock_s2_get.return_value = s2_response

        oa_response = MagicMock()
        oa_response.status_code = 200
        oa_response.json.return_value = OA_SEARCH_RESPONSE
        mock_oa_get.return_value = oa_response

        query_plan = _make_query_plan("navigational")
        papers, edges = retrieve_live_papers(query_plan)

        self.assertGreater(len(papers), 0)
        self.assertIsInstance(papers[0], Paper)


if __name__ == "__main__":
    unittest.main()