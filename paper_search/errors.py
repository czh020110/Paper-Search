from __future__ import annotations

import httpx


class PaperSearchError(Exception):
    """Base error for all paper_search errors."""


class ApiTimeoutError(PaperSearchError):
    """Request to an external API timed out."""


class ApiRateLimitError(PaperSearchError):
    """External API returned a rate-limit response (429)."""

    def __init__(self, message: str = "", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ApiError(PaperSearchError):
    """External API returned a non-success HTTP status."""

    def __init__(self, message: str = "", status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ParseError(PaperSearchError):
    """Failed to parse or validate structured data."""


class ValidationError(PaperSearchError):
    """Input or contract validation failed."""


class BudgetExceededError(PaperSearchError):
    """A hard budget limit was reached."""


class LlmError(PaperSearchError):
    """LLM invocation or output parsing failed."""


def classify_httpx_error(error: Exception) -> PaperSearchError:
    """Map an httpx exception to the appropriate :class:`PaperSearchError` subclass."""
    if isinstance(error, httpx.TimeoutException):
        return ApiTimeoutError(str(error))
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code  # type: ignore[union-attr]
        if status == 429:
            retry_after = error.response.headers.get("retry-after")  # type: ignore[union-attr]
            seconds = float(retry_after) if retry_after else None
            return ApiRateLimitError(str(error), retry_after=seconds)
        return ApiError(str(error), status_code=status)
    if isinstance(error, httpx.RequestError):
        return ApiError(str(error))
    return PaperSearchError(str(error))


def is_retryable(error: PaperSearchError | Exception) -> bool:
    """Return ``True`` if the error is transient and worth retrying."""
    if isinstance(error, ApiTimeoutError):
        return True
    if isinstance(error, ApiRateLimitError):
        return True
    if isinstance(error, ApiError) and error.status_code is not None and error.status_code >= 500:
        return True
    return False