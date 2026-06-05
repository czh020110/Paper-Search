from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from paper_search.cache import CacheStore


class TestCacheStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.cache = CacheStore(self.tmpdir)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_put_and_get(self) -> None:
        self.cache.put("api_responses", "https://api.example.com/v1/papers?q=test", data={"results": [1, 2, 3]})
        result = self.cache.get("api_responses", "https://api.example.com/v1/papers?q=test")
        self.assertEqual(result, {"results": [1, 2, 3]})

    def test_miss_returns_none(self) -> None:
        result = self.cache.get("api_responses", "nonexistent_key")
        self.assertIsNone(result)

    def test_cache_key_does_not_contain_api_key(self) -> None:
        # The cache key is a SHA-256 hash; verify no API key leaks into filenames
        url_with_key = "https://api.example.com/v1/papers?q=test&api_key=SECRET123"
        self.cache.put("api_responses", url_with_key, data={"ok": True})
        # Scan all files in cache dir — no file should contain the API key in plaintext
        for f in (self.tmpdir / "api_responses").rglob("*"):
            if f.is_file() and f.suffix == ".json":
                content = f.read_text(encoding="utf-8")
                self.assertNotIn("SECRET123", content)

    def test_ttl_expiry(self) -> None:
        self.cache.put("api_responses", "ttl_test", data={"temp": True}, ttl_seconds=1)
        result = self.cache.get("api_responses", "ttl_test")
        self.assertIsNotNone(result)
        time.sleep(1.1)
        expired = self.cache.get("api_responses", "ttl_test")
        self.assertIsNone(expired)

    def test_no_ttl_does_not_expire(self) -> None:
        self.cache.put("embeddings", "permanent_key", data={"vec": [0.1, 0.2]}, ttl_seconds=None)
        result = self.cache.get("embeddings", "permanent_key")
        self.assertIsNotNone(result)

    def test_version_invalidation(self) -> None:
        self.cache.put("llm_judgments", "prompt_v1", data={"score": 8}, version="v1")
        self.cache.put("llm_judgments", "prompt_v1_extra", data={"score": 7}, version="v1")
        self.cache.put("llm_judgments", "prompt_v2", data={"score": 9}, version="v2")

        removed = self.cache.invalidate("llm_judgments", "v2")
        self.assertEqual(removed, 2)  # v1 entries removed, v2 stays

        self.assertIsNone(self.cache.get("llm_judgments", "prompt_v1"))
        self.assertIsNone(self.cache.get("llm_judgments", "prompt_v1_extra"))
        self.assertIsNotNone(self.cache.get("llm_judgments", "prompt_v2"))

    def test_domain_subdirectories_created(self) -> None:
        for domain in ("paper_metadata", "citations", "embeddings", "llm_judgments", "api_responses"):
            self.assertTrue((self.tmpdir / domain).is_dir())

    def test_stats_and_hit_rate(self) -> None:
        self.cache.put("api_responses", "hit_key", data={"ok": True})
        self.cache.get("api_responses", "hit_key")  # hit
        self.cache.get("api_responses", "miss_key")  # miss
        stats = self.cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(self.cache.hit_rate(), 0.5)

    def test_same_key_different_domains(self) -> None:
        self.cache.put("paper_metadata", "paper_1", data={"title": "A"})
        self.cache.put("citations", "paper_1", data={"refs": ["B", "C"]})
        meta = self.cache.get("paper_metadata", "paper_1")
        cites = self.cache.get("citations", "paper_1")
        self.assertEqual(meta, {"title": "A"})
        self.assertEqual(cites, {"refs": ["B", "C"]})


if __name__ == "__main__":
    unittest.main()
