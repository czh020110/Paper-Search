from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import httpx

from paper_search.errors import (
    ApiError,
    ApiRateLimitError,
    ApiTimeoutError,
    BudgetExceededError,
    LlmError,
    PaperSearchError,
    ParseError,
    ValidationError,
    classify_httpx_error,
    is_retryable,
)


class TestErrorHierarchy(unittest.TestCase):
    def test_all_errors_inherit_from_base(self) -> None:
        for cls in (ApiTimeoutError, ApiRateLimitError, ApiError, ParseError, ValidationError, BudgetExceededError, LlmError):
            self.assertTrue(issubclass(cls, PaperSearchError))

    def test_api_rate_limit_has_retry_after(self) -> None:
        err = ApiRateLimitError("rate limited", retry_after=5.0)
        self.assertEqual(err.retry_after, 5.0)

    def test_api_error_has_status_code(self) -> None:
        err = ApiError("server error", status_code=500)
        self.assertEqual(err.status_code, 500)


class TestClassifyHttpxError(unittest.TestCase):
    def test_timeout_maps_to_api_timeout(self) -> None:
        err = httpx.TimeoutException("timed out")
        result = classify_httpx_error(err)
        self.assertIsInstance(result, ApiTimeoutError)

    def test_429_maps_to_rate_limit(self) -> None:
        response = MagicMock()
        response.status_code = 429
        response.headers = {}
        err = httpx.HTTPStatusError("429", request=MagicMock(), response=response)
        result = classify_httpx_error(err)
        self.assertIsInstance(result, ApiRateLimitError)

    def test_429_extracts_retry_after(self) -> None:
        response = MagicMock()
        response.status_code = 429
        response.headers = {"retry-after": "10"}
        err = httpx.HTTPStatusError("429", request=MagicMock(), response=response)
        result = classify_httpx_error(err)
        self.assertIsInstance(result, ApiRateLimitError)
        self.assertEqual(result.retry_after, 10.0)  # type: ignore[union-attr]

    def test_500_maps_to_api_error(self) -> None:
        response = MagicMock()
        response.status_code = 500
        err = httpx.HTTPStatusError("500", request=MagicMock(), response=response)
        result = classify_httpx_error(err)
        self.assertIsInstance(result, ApiError)
        self.assertEqual(result.status_code, 500)  # type: ignore[union-attr]

    def test_404_maps_to_api_error(self) -> None:
        response = MagicMock()
        response.status_code = 404
        err = httpx.HTTPStatusError("404", request=MagicMock(), response=response)
        result = classify_httpx_error(err)
        self.assertIsInstance(result, ApiError)
        self.assertEqual(result.status_code, 404)  # type: ignore[union-attr]

    def test_request_error_maps_to_api_error(self) -> None:
        err = httpx.ConnectError("connection refused")
        result = classify_httpx_error(err)
        self.assertIsInstance(result, ApiError)

    def test_unknown_error_maps_to_base(self) -> None:
        result = classify_httpx_error(ValueError("oops"))
        self.assertIsInstance(result, PaperSearchError)
        self.assertNotIsInstance(result, ApiTimeoutError)


class TestIsRetryable(unittest.TestCase):
    def test_timeout_is_retryable(self) -> None:
        self.assertTrue(is_retryable(ApiTimeoutError("timeout")))

    def test_rate_limit_is_retryable(self) -> None:
        self.assertTrue(is_retryable(ApiRateLimitError("429")))

    def test_5xx_is_retryable(self) -> None:
        self.assertTrue(is_retryable(ApiError("server error", status_code=500)))
        self.assertTrue(is_retryable(ApiError("bad gateway", status_code=502)))

    def test_4xx_is_not_retryable(self) -> None:
        self.assertFalse(is_retryable(ApiError("not found", status_code=404)))
        self.assertFalse(is_retryable(ApiError("bad request", status_code=400)))

    def test_parse_error_is_not_retryable(self) -> None:
        self.assertFalse(is_retryable(ParseError("bad json")))

    def test_budget_exceeded_is_not_retryable(self) -> None:
        self.assertFalse(is_retryable(BudgetExceededError("limit reached")))


if __name__ == "__main__":
    unittest.main()
