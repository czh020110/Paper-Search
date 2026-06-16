from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler

from paper_search.llm import get_structured_output_llm, get_token_callback


class TestLlmFactory(unittest.TestCase):
    @patch("langchain_openai.ChatOpenAI")
    def test_structured_output_llm_uses_langchain_callback_handler(self, mock_chat_openai) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "LLM_FAST_MODEL": "gpt-4o-mini",
                "LLM_THINKING": "high",
            },
            clear=False,
        ):
            get_structured_output_llm(temperature=0.0)
            self.assertEqual(os.environ.get("LLM_THINKING"), "high")

        kwargs = mock_chat_openai.call_args.kwargs
        callbacks = kwargs["callbacks"]

        self.assertEqual(len(callbacks), 1)
        self.assertIs(callbacks[0], get_token_callback())
        self.assertIsInstance(callbacks[0], BaseCallbackHandler)

    def test_structured_output_llm_instantiates_without_callback_validation_error(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "LLM_FAST_MODEL": "gpt-4o-mini",
                "LLM_THINKING": "high",
            },
            clear=False,
        ):
            llm = get_structured_output_llm(temperature=0.0)
            self.assertEqual(os.environ.get("LLM_THINKING"), "high")

        self.assertEqual(type(llm).__name__, "ChatOpenAI")


if __name__ == "__main__":
    unittest.main()
