from __future__ import annotations

import unittest

from paper_search.budget import BudgetController, BudgetLimits


class TestBudgetController(unittest.TestCase):
    def test_unlimited_by_default(self) -> None:
        limits = BudgetLimits()
        budget = BudgetController(limits)
        self.assertTrue(budget.can_call_api())
        self.assertTrue(budget.can_use_tokens(999999))
        self.assertTrue(budget.can_expand_pool(1000, 1000))
        self.assertTrue(budget.can_start_round())

    def test_api_call_limit_enforced(self) -> None:
        limits = BudgetLimits(max_api_calls=3)
        budget = BudgetController(limits)
        self.assertTrue(budget.can_call_api())
        budget.increment_api_calls()
        budget.increment_api_calls()
        budget.increment_api_calls()
        self.assertFalse(budget.can_call_api())

    def test_token_limit_enforced(self) -> None:
        limits = BudgetLimits(max_tokens=100)
        budget = BudgetController(limits)
        self.assertTrue(budget.can_use_tokens(50))
        budget.record_tokens(60)
        self.assertFalse(budget.can_use_tokens(50))
        self.assertTrue(budget.can_use_tokens(30))

    def test_pool_size_limit_enforced(self) -> None:
        limits = BudgetLimits(max_pool_size=10)
        budget = BudgetController(limits)
        self.assertTrue(budget.can_expand_pool(8, 2))
        self.assertFalse(budget.can_expand_pool(9, 2))

    def test_round_limit_enforced(self) -> None:
        limits = BudgetLimits(max_rounds=2)
        budget = BudgetController(limits)
        self.assertTrue(budget.can_start_round())
        budget.increment_round()
        budget.increment_round()
        self.assertFalse(budget.can_start_round())

    def test_none_means_unlimited(self) -> None:
        limits = BudgetLimits(max_api_calls=None, max_tokens=None, max_rounds=None, max_pool_size=None)
        budget = BudgetController(limits)
        budget.increment_api_calls()
        budget.increment_api_calls()
        budget.record_tokens(999999)
        budget.increment_round()
        self.assertTrue(budget.can_call_api())
        self.assertTrue(budget.can_use_tokens(1))
        self.assertTrue(budget.can_expand_pool(1000, 1000))
        self.assertTrue(budget.can_start_round())

    def test_usage_summary_structure(self) -> None:
        limits = BudgetLimits(max_api_calls=10, max_tokens=1000, max_rounds=3, max_pool_size=50)
        budget = BudgetController(limits)
        budget.increment_api_calls()
        budget.increment_api_calls()
        budget.record_tokens(200)
        summary = budget.usage_summary()
        self.assertEqual(summary["api_calls_used"], 2)
        self.assertEqual(summary["tokens_used"], 200)
        self.assertEqual(summary["current_round"], 0)
        self.assertEqual(summary["limits"]["max_api_calls"], 10)
        self.assertEqual(summary["limits"]["max_tokens"], 1000)
        self.assertEqual(summary["limits"]["max_pool_size"], 50)


if __name__ == "__main__":
    unittest.main()