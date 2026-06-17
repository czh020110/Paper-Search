from __future__ import annotations

import unittest

from paper_search.retrieval.live_backend import _normalize_oa_sort


class TestLiveBackendHelpers(unittest.TestCase):
    def test_normalize_oa_sort_maps_legacy_citation_count(self) -> None:
        self.assertEqual(_normalize_oa_sort("citation_count:desc"), "cited_by_count:desc")
        self.assertEqual(_normalize_oa_sort("citation_count:asc"), "cited_by_count:asc")

    def test_normalize_oa_sort_preserves_valid_sort(self) -> None:
        self.assertEqual(_normalize_oa_sort("relevance_score:desc"), "relevance_score:desc")
        self.assertEqual(_normalize_oa_sort("publication_date:desc"), "publication_date:desc")


if __name__ == "__main__":
    unittest.main()
