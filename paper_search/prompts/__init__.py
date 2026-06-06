from __future__ import annotations

# Prompt registry — maps prompt_name to metadata and template.
# Each entry stores: name, version, hash placeholder, expected output schema,
# and the raw template string.

QUERY_UNDERSTANDING_PROMPT = """\
You are an expert academic research assistant. Given a natural language query about academic papers, produce a structured JSON output that extracts and expands the query into a format suitable for academic paper search APIs.

## Rules

1. **intent_analysis.query_type**: Classify as one of:
   - "navigational": The user is looking for a SPECIFIC paper (mentions title, author, or exact paper name).
   - "semantic": The user wants papers about a METHOD/TOPIC (broad research area).
   - "metadata": The user gives structured constraints like year, venue, author (without strong topic keywords).

2. **intent_analysis.domain**: Be specific, e.g. "Computer Vision / Natural Language Processing (Vision-Language Models)", not "Academic Search".

3. **hard_filters.year**: Extract explicit year constraints from the query. Use ">=" operator. Include a relaxed_window.

4. **ranking_signals.preferred_venues**: Extract venue mentions (CVPR, NeurIPS, ICLR, ACL, EMNLP, etc.) and include their aliases. Set venue_as_hard_filter to false.

5. **semantic_queries**: Generate TWO groups of English keywords:
   - core_concepts: topic/concept keywords (e.g. "large language model", "hallucination", "classical chinese")
   - methodologies: method/technique keywords (e.g. "reinforcement learning", "RLHF", "jailbreak")

6. **sub_queries_for_retrieval**: Build 3-5 English search queries by combining the keywords above. ALL output must be in English — NEVER include Chinese, Japanese, or Korean characters in queries.

7. **query_expansion_policy**: Always include with enabled=true, seed_paper_driven=true, extract_terms_from=["title","abstract","keywords"], max_rounds=2.

8. **api_payload_translation**: Translate sub_queries into API-specific payloads:
   - semantic_scholar: {{"query": <sub_query>, "year": "<year>-"}}
   - openalex: {{"search": <sub_query>, "filter": "publication_year:><year-1>"}}

## User Query

{query}

## Output Format

Return ONLY a valid JSON object with exactly this structure. No extra text.
"""

QUERY_UNDERSTANDING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "intent_analysis": {
            "type": "object",
            "properties": {
                "domain": {"type": "string"},
                "query_type": {"type": "string", "enum": ["navigational", "semantic", "metadata"]},
                "boundary_note": {"type": "string"},
            },
            "required": ["domain", "query_type", "boundary_note"],
        },
        "hard_filters": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "object",
                    "properties": {
                        "operator": {"type": "string"},
                        "value": {"type": "integer"},
                        "relaxed_window": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["operator", "value", "relaxed_window"],
                }
            },
            "required": ["year"],
        },
        "ranking_signals": {
            "type": "object",
            "properties": {
                "preferred_venues": {"type": "array", "items": {"type": "string"}},
                "venue_match_mode": {"enum": ["fuzzy_match_and_bonus"]},
                "venue_as_hard_filter": {"type": "boolean"},
            },
            "required": ["preferred_venues", "venue_match_mode", "venue_as_hard_filter"],
        },
        "semantic_queries": {
            "type": "object",
            "properties": {
                "core_concepts": {"type": "array", "items": {"type": "string"}},
                "methodologies": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["core_concepts", "methodologies"],
        },
        "sub_queries_for_retrieval": {"type": "array", "items": {"type": "string"}},
        "query_expansion_policy": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "seed_paper_driven": {"type": "boolean"},
                "extract_terms_from": {"type": "array", "items": {"type": "string"}},
                "max_rounds": {"type": "integer"},
            },
            "required": ["enabled", "seed_paper_driven", "extract_terms_from", "max_rounds"],
        },
        "api_payload_translation": {
            "type": "object",
            "properties": {
                "semantic_scholar": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "year": {"type": "string"}},
                    },
                },
                "openalex": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"search": {"type": "string"}, "filter": {"type": "string"}},
                    },
                },
            },
            "required": ["semantic_scholar", "openalex"],
        },
    },
    "required": [
        "intent_analysis",
        "hard_filters",
        "ranking_signals",
        "semantic_queries",
        "sub_queries_for_retrieval",
        "query_expansion_policy",
        "api_payload_translation",
    ],
}

JUDGE_PROMPT = """\
角色：严谨的学术领域专家。根据【原始学术查询】评估以下【候选论文列表】的相关性。

【用户原始学术查询】：
{query}

【候选论文列表】：
{papers_json}

【任务指令】：
1. 对比查询意图与论文内容，判断是否满足主题、方法、时间与 venue 要求。
2. venue 字段缺失或写法不一致时，不直接判负，需结合标题、摘要与其他元数据综合判断。
3. 分类结果只能在 [高度相关, 部分相关, 不相关] 中选择。
4. 输出一句判定理由与一句主要贡献。
5. 严格输出 JSON 数组，不添加额外说明。

输出格式：
[
  {{
    "paper_id": "<传入的论文ID>",
    "relevance": "<高度相关/部分相关/不相关>",
    "reason": "<判定理由（一句）>",
    "contribution": "<论文主要贡献（一句）>"
  }}
]
"""

JUDGE_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string"},
            "relevance": {"type": "string", "enum": ["高度相关", "部分相关", "不相关"]},
            "reason": {"type": "string"},
            "contribution": {"type": "string"},
        },
        "required": ["paper_id", "relevance", "reason", "contribution"],
    },
}

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

RERANK_PROMPT = """\
You are an academic relevance scorer. Given a user query and a list of candidate papers,
score each paper's relevance to the query on a 0-10 scale.

Scoring guidelines:
- 8-10: Directly addresses the query's core topic and methodology
- 5-7: Partially relevant — shares topic or method but not both
- 1-4: Tangentially related
- 0: Not relevant

【用户查询】：
{query}

【候选论文列表】：
{papers_json}

Return ONLY a JSON array of objects with paper_id and score:
[{{"paper_id": "<id>", "score": <0-10>}}]
"""

RERANK_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string"},
            "score": {"type": "number", "minimum": 0, "maximum": 10},
        },
        "required": ["paper_id", "score"],
    },
}

PROMPT_REGISTRY: dict[str, dict] = {
    "query_understanding": {
        "name": "query_understanding",
        "version": "v1",
        "description": "Convert natural language academic query to structured QueryPlan JSON",
        "template": QUERY_UNDERSTANDING_PROMPT,
        "expected_schema": QUERY_UNDERSTANDING_SCHEMA,
    },
    "judge": {
        "name": "judge",
        "version": "v1",
        "description": "Batch relevance judgment: 高度相关/部分相关/不相关 + reason + contribution",
        "template": JUDGE_PROMPT,
        "expected_schema": JUDGE_SCHEMA,
    },
    "rerank": {
        "name": "rerank",
        "version": "v1",
        "description": "LLM-based pairwise relevance scoring (0-10 scale)",
        "template": RERANK_PROMPT,
        "expected_schema": RERANK_SCHEMA,
    },
}
