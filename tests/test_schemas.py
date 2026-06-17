from __future__ import annotations

import unittest
from typing import Any, cast

from paper_search.schemas import S2PayloadSchema


class TestSchemas(unittest.TestCase):
    def test_s2_payload_accepts_fields_of_study_list(self) -> None:
        payload = S2PayloadSchema(
            query="hallucination mitigation in large language models",
            fields_of_study=cast(Any, ["Computer Science", "Artificial Intelligence"]),
        )

        self.assertEqual(payload.fields_of_study, "Computer Science,Artificial Intelligence")


if __name__ == "__main__":
    unittest.main()
