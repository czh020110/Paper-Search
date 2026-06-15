from __future__ import annotations

# Prompt registry — maps prompt_name to metadata and template.
# Each entry stores: name, version, hash placeholder, expected output schema,
# and the raw template string.

QUERY_UNDERSTANDING_PROMPT = """\
You are an expert academic research assistant. Given a natural language query about academic papers, produce a structured output that extracts and expands the query into a format suitable for academic paper search APIs.

## Critical Rules

1. ALL text fields intended for search APIs must be in English (ASCII only). This includes core_concepts, methodologies, sub_queries_for_retrieval, and all fields inside api_payload_translation. NEVER output Chinese, Japanese, or Korean characters in these fields.

2. semantic_scholar and openalex inside api_payload_translation are INDEPENDENT — they can have DIFFERENT counts and DIFFERENT search strategies. Do NOT try to make them match in count or order.

3. S2 (Semantic Scholar) is a SEMANTIC search engine:
   - The `query` field supports natural language phrases — long descriptive sentences work WELL.
   - Use AND/OR/NOT for precise logical combinations: "hallucination AND (reinforcement learning OR RLHF)".
   - Leverage structured filter parameters: `venue` for venue filtering, `fieldsOfStudy` for domain, `minCitationCount` for quality, `year` for time range.
   - Strategy: generate 1-3 RICH payloads that combine multiple concepts into semantic queries with filter narrowing.

4. OpenAlex uses BM25 keyword matching (NOT semantic search):
   - Words in `search` are ANDed — every word must appear in a matching paper.
   - Compose a CONCISE KEYWORD SENTENCE with ALL relevant non-redundant terms, excluding hard constraints (year/venue/author go in filter).
   - Good: "hallucination mitigation large language model reinforcement learning RLHF CVPR" — all key concepts + venue name present, no filler words.
   - Bad: "how to mitigate hallucination in large language models using reinforcement learning" — too semantic/natural-language for BM25.
   - Bad: "hallucination reinforcement learning" — too few keywords, too broad (unless the query really only has 2 concepts).
   - Use exact phrases for compound terms: '"object hallucination"'.
   - Use `filter` fields for year and type precision. For venue filtering, include the venue name in `search` — do NOT use `primary_location_source_id` unless you KNOW the exact OpenAlex source ID (do NOT guess or fabricate IDs).
   - Strategy: generate 1-3 payloads covering different search angles.

5. sub_queries_for_retrieval is a search intent SUMMARY — it is NOT sent to APIs directly. It serves as: (a) a human-readable summary for result display, (b) reranker query input, (c) snowball expansion deduplication seed.

6. arXiv supports STRUCTURED field-prefix search:
   - Use ti: for title terms, abs: for abstract terms, cat: for category (e.g. cs.CV, cs.CL, cs.AI, cs.LG), au: for author.
   - Use AND/OR/ANDNOT for boolean logic.
   - Good: "ti:hallucination AND abs:reinforcement learning AND cat:cs.CV" — precise field targeting.
   - Bad: "all:hallucination reinforcement learning" — too broad, no field precision.
   - Use submitted_date_from/to for year filtering (YYYYMMDD format, e.g. 20220101).
   - arXiv excels at finding recent preprints — generate 1-2 payloads targeting categories relevant to the query. Common category mappings: Computer Vision → cs.CV, NLP → cs.CL, AI → cs.AI, ML → cs.LG, Robotics → cs.RO.
   - If the query mentions a specific venue (e.g. CVPR, ACL), include the venue name as an abs: or ti: term AND set the category accordingly — do NOT use a venue filter field (arXiv has none).

7. Follow the field descriptions in the function schema for detailed guidance on what each field should contain.

## User Query

{query}
"""

# 清理说明：JUDGE_PROMPT/JUDGE_SCHEMA、RERANK_PROMPT/RERANK_SCHEMA 及 PROMPT_REGISTRY
# 已无任何模块引用，予以删除。仅保留 JUDGE_SINGLE_*（被 screening/judge.py 使用）。

# Single-paper version of judge prompt for concurrent per-paper LLM calls.
JUDGE_SINGLE_PROMPT = """\
角色：严谨的学术领域专家。根据【原始学术查询】评估下面这篇论文的相关性。

【原始学术查询】：
{query}

【论文信息】：
- 标题：{title}
- 摘要：{abstract}
- Venue：{venue}
- 年份：{year}

【任务指令】：
1. 对比查询意图与论文内容，判断是否满足主题、方法、时间与 venue 要求。
2. venue 字段缺失或写法不一致时，不直接判负，需结合标题、摘要与其他元数据综合判断。
3. 分类结果只能在 [高度相关, 部分相关, 不相关] 中选择。
4. 输出一句判定理由与一句主要贡献。
5. 严格输出 JSON 对象，不添加额外说明。

输出格式：
{{
  "paper_id": "{paper_id}",
  "relevance": "<高度相关/部分相关/不相关>",
  "reason": "<判定理由（一句）>",
  "contribution": "<论文主要贡献（一句）>"
}}
"""

JUDGE_SINGLE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "paper_id": {"type": "string"},
        "relevance": {"type": "string", "enum": ["高度相关", "部分相关", "不相关"]},
        "reason": {"type": "string"},
        "contribution": {"type": "string"},
    },
    "required": ["paper_id", "relevance", "reason", "contribution"],
}

JUDGE_CONCURRENCY = 100
