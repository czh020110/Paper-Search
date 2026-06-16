from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from paper_search.embedding import _DASHSCOPE_BATCH_SIZE, batch_embeddings


class TestEmbeddingBatching(unittest.TestCase):
    @patch("dashscope.TextEmbedding.call")
    def test_dashscope_batch_embeddings_split_into_provider_limit(self, mock_call: MagicMock) -> None:
        def _make_response(batch: list[str]) -> MagicMock:
            response = MagicMock()
            response.status_code = 200
            response.output = {
                "embeddings": [
                    {"embedding": [float(index), float(index) + 0.5]}
                    for index, _ in enumerate(batch)
                ]
            }
            return response

        def _side_effect(*, model: str, input: list[str], api_key: str) -> MagicMock:
            self.assertLessEqual(len(input), _DASHSCOPE_BATCH_SIZE)
            return _make_response(input)

        mock_call.side_effect = _side_effect

        texts = [f"paper {i}" for i in range(_DASHSCOPE_BATCH_SIZE * 2 + 3)]
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_PROVIDER": "dashscope",
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_MODEL": "text-embedding-v4",
            },
            clear=False,
        ):
            results = batch_embeddings(texts)

        self.assertEqual(len(results), len(texts))
        self.assertEqual(mock_call.call_count, 3)
        self.assertTrue(all(result is not None for result in results))

        batch_sizes = [call.kwargs["input"] for call in mock_call.call_args_list]
        self.assertEqual([len(batch) for batch in batch_sizes], [10, 10, 3])


if __name__ == "__main__":
    unittest.main()
