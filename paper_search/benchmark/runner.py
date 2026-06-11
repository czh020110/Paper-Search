"""评测运行器：跑搜索管线 + 算分。"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

from .dataset import load_dataset, download_normalizer

logger = logging.getLogger(__name__)


def _get_corpus_ids_from_pipeline(papers: list[dict]) -> list[str]:
    """从管线结果提取 S2 corpus_id 列表。

    对 S2 来源直接从 raw.externalIds.CorpusId 取；
    对 DBLP/OpenAlex 来源，批量调 S2 paper search API 用标题查 corpus_id。
    """
    ids: list[str] = []
    need_lookup: list[tuple[int, str]] = []  # (index, title) for batch S2 lookup

    for i, p in enumerate(papers):
        # 1) direct field
        cid = p.get("corpusId") or p.get("corpus_id")
        if cid:
            ids.append(str(cid))
            continue
        # 2) raw.externalIds.CorpusId (S2 source)
        raw = p.get("raw") or {}
        ext = raw.get("externalIds") or {}
        cid = ext.get("CorpusId")
        if cid:
            ids.append(str(cid))
            continue
        # 3) paperId is numeric corpus_id
        pid = p.get("paperId") or ""
        if pid and pid.isdigit():
            ids.append(pid)
            continue
        # 4) defer to batch lookup
        title = p.get("title", "")
        if title:
            need_lookup.append((i, title))
        ids.append("")  # placeholder

    if need_lookup:
        logger.info("Batch-looking up %d corpus_ids via S2 API", len(need_lookup))
        _s2_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("ASTA_TOOL_KEY")
        _headers: dict[str, str] = {"Accept": "application/json"}
        if _s2_api_key:
            _headers["x-api-key"] = _s2_api_key

        import httpx
        with httpx.Client(trust_env=False, timeout=10.0) as client:
            for idx, title in need_lookup:
                try:
                    resp = client.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        params={
                            "query": title[:200],
                            "limit": 1,
                            "fields": "corpusId",
                        },
                        headers=_headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", [])
                        if data and data[0].get("corpusId"):
                            ids[idx] = str(data[0]["corpusId"])
                            logger.debug("S2 lookup: '%s' → corpusId=%s", title[:40], ids[idx])
                except Exception as e:
                    logger.warning("S2 lookup failed for '%s': %s", title[:40], e)

    return [x for x in ids if x]


def _score_specific(
    predicted_ids: list[str],
    ground_truth_ids: list[str],
) -> dict[str, float]:
    """计算 specific/metadata 类型的 recall / precision / F1。"""
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    pred_set = set(predicted_ids)
    hits = pred_set & gt_set

    recall = len(hits) / len(gt_set)
    precision = len(hits) / len(pred_set) if pred_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"recall": recall, "precision": precision, "f1": f1}


def _score_recall_at_k(
    predicted_ids: list[str],
    ground_truth_ids: list[str],
    k: int,
) -> float:
    """计算 recall@K。"""
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    top_k = set(predicted_ids[:k])
    return len(top_k & gt_set) / len(gt_set)


def run_benchmark(
    on_progress: Callable[[dict], None] | None = None,
    limit: int = 0,
    skip_semantic: bool = True,
) -> dict[str, Any]:
    """跑完整评测。

    Args:
        on_progress: 进度回调，每跑完一条调用
        limit: 限制跑几条（0=全部）
        skip_semantic: 跳过 semantic 类型（需要 LLM 评判）

    Returns:
        汇总结果 dict
    """
    from ..pipeline import run_pipeline
    from ..config import load_settings
    from pathlib import Path as _Path
    _settings = load_settings()
    _output_root = _Path(_settings.output_dir)
    _output_root.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    normalizer = download_normalizer()

    if skip_semantic:
        data = [s for s in data if not s["input"]["query_id"].startswith("semantic")]
    if limit > 0:
        data = data[:limit]

    total = len(data)
    results: list[dict] = []
    all_metrics: dict[str, list[float]] = {"recall": [], "precision": [], "f1": []}

    for i, sample in enumerate(data):
        query_id = sample["input"]["query_id"]
        query = sample["input"]["query"]
        qtype = query_id.split("_")[0]
        scorer = sample["scorer_criteria"]

        # 即时反馈：正在处理
        if on_progress:
            on_progress({
                "current": i + 1,
                "total": total,
                "entry": {"query_id": query_id, "type": qtype, "status": "running"},
                "running_avg": {k: round(sum(v) / len(v), 4) for k, v in all_metrics.items() if v} or {"recall": 0},
            })

        t0 = time.time()
        try:
            pipeline_result = run_pipeline(query=query, backend="live", output_root=_output_root, settings=_settings)
            papers = pipeline_result.get("papers", [])
            predicted_ids = _get_corpus_ids_from_pipeline(papers)
        except Exception as e:
            logger.error("Pipeline failed for %s: %s", query_id, e)
            predicted_ids = []
        elapsed = time.time() - t0

        # 评分
        if qtype in ("specific", "metadata") and "corpus_ids" in scorer:
            gt_ids = scorer["corpus_ids"]
            metrics = _score_specific(predicted_ids, gt_ids)
        elif qtype == "litqa2" and "corpus_ids" in scorer:
            gt_ids = scorer["corpus_ids"]
            metrics = {
                "recall": _score_recall_at_k(predicted_ids, gt_ids, 30),
                "precision": 0.0,
                "f1": 0.0,
            }
        else:
            metrics = {"recall": 0.0, "precision": 0.0, "f1": 0.0}

        entry = {
            "query_id": query_id,
            "query": query[:80],
            "type": qtype,
            "elapsed": round(elapsed, 1),
            "n_predicted": len(predicted_ids),
            **metrics,
        }
        results.append(entry)

        for k in ("recall", "precision", "f1"):
            all_metrics[k].append(metrics.get(k, 0.0))

        if on_progress:
            on_progress({
                "current": i + 1,
                "total": total,
                "entry": entry,
                "running_avg": {
                    k: round(sum(v) / len(v), 4) for k, v in all_metrics.items() if v
                },
            })

    # 汇总
    summary = {
        "total": total,
        "avg_recall": round(sum(all_metrics["recall"]) / max(len(all_metrics["recall"]), 1), 4),
        "avg_precision": round(sum(all_metrics["precision"]) / max(len(all_metrics["precision"]), 1), 4),
        "avg_f1": round(sum(all_metrics["f1"]) / max(len(all_metrics["f1"]), 1), 4),
        "results": results,
    }
    return summary
