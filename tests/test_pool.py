from __future__ import annotations

import unittest

from paper_search.contracts import Paper
from paper_search.pool import CandidatePool, InvalidTransition


def _make_paper(paper_id: str = "test:1", title: str = "Test Paper") -> Paper:
    return Paper(
        id=paper_id,
        source_ids={"doi": None},
        title=title,
        abstract=None,
        pool_status="discovered",
    )


class TestCandidatePool(unittest.TestCase):
    def test_add_with_default_status(self) -> None:
        pool = CandidatePool()
        paper = _make_paper()
        pool.add(paper)
        self.assertEqual(paper.pool_status, "discovered")

    def test_add_with_explicit_status(self) -> None:
        pool = CandidatePool()
        paper = _make_paper()
        pool.add(paper, status="seed")
        self.assertEqual(paper.pool_status, "seed")

    def test_valid_transition(self) -> None:
        pool = CandidatePool()
        paper = _make_paper("p1")
        pool.add(paper, status="seed")
        pool.transition("p1", "rough_scored", reason="coarse_screening")
        self.assertEqual(paper.pool_status, "rough_scored")

    def test_invalid_transition_raises(self) -> None:
        pool = CandidatePool()
        paper = _make_paper("p1")
        pool.add(paper, status="seed")
        with self.assertRaises(InvalidTransition):
            pool.transition("p1", "llm_judged")  # skip rough_scored and reranked

    def test_transition_to_excluded_from_any_nonterminal(self) -> None:
        pool = CandidatePool()
        for status in ("discovered", "seed", "expanded", "rough_scored", "reranked", "llm_judged"):
            paper = _make_paper(f"p_{status}")
            pool.add(paper, status=status)
            pool.transition(f"p_{status}", "excluded", reason="irrelevant")
            self.assertEqual(paper.pool_status, "excluded")

    def test_terminal_status_cannot_transition(self) -> None:
        pool = CandidatePool()
        paper = _make_paper("p1")
        pool.add(paper, status="selected")
        with self.assertRaises(InvalidTransition):
            pool.transition("p1", "seed")

    def test_transition_all(self) -> None:
        pool = CandidatePool()
        for i in range(5):
            pool.add(_make_paper(f"p{i}"), status="seed")
        count = pool.transition_all("seed", "selected", reason="minimal_loop")
        self.assertEqual(count, 5)
        self.assertEqual(len(pool.by_status("selected")), 5)

    def test_transition_all_only_affects_matching_status(self) -> None:
        pool = CandidatePool()
        pool.add(_make_paper("p1"), status="seed")
        pool.add(_make_paper("p2"), status="expanded")
        count = pool.transition_all("seed", "rough_scored")
        self.assertEqual(count, 1)
        p1 = pool.get("p1")
        p2 = pool.get("p2")
        assert p1 is not None
        assert p2 is not None
        self.assertEqual(p1.pool_status, "rough_scored")
        self.assertEqual(p2.pool_status, "expanded")

    def test_by_status(self) -> None:
        pool = CandidatePool()
        pool.add(_make_paper("p1"), status="seed")
        pool.add(_make_paper("p2"), status="seed")
        pool.add(_make_paper("p3"), status="excluded")
        self.assertEqual(len(pool.by_status("seed")), 2)
        self.assertEqual(len(pool.by_status("excluded")), 1)

    def test_summary(self) -> None:
        pool = CandidatePool()
        pool.add(_make_paper("p1"), status="seed")
        pool.add(_make_paper("p2"), status="seed")
        pool.add(_make_paper("p3"), status="excluded")
        summary = pool.summary()
        self.assertEqual(summary, {"seed": 2, "excluded": 1})

    def test_len(self) -> None:
        pool = CandidatePool()
        for i in range(3):
            pool.add(_make_paper(f"p{i}"))
        self.assertEqual(len(pool), 3)

    def test_get_nonexistent_raises_keyerror(self) -> None:
        pool = CandidatePool()
        with self.assertRaises(KeyError):
            pool.transition("nonexistent", "seed")

    def test_invalid_initial_status_raises(self) -> None:
        pool = CandidatePool()
        with self.assertRaises(InvalidTransition):
            pool.add(_make_paper(), status="not_a_status")

    def test_invalid_target_status_raises(self) -> None:
        pool = CandidatePool()
        pool.add(_make_paper("p1"), status="seed")
        with self.assertRaises(InvalidTransition):
            pool.transition("p1", "not_a_status")


if __name__ == "__main__":
    unittest.main()
