from __future__ import annotations

# Prompt registry — maps prompt_name to metadata and template.
# Each entry stores: name, version, hash placeholder, expected output schema,
# and the raw template string.

QUERY_UNDERSTANDING_PROMPT = """\
You are an expert academic research assistant. Given a natural language query about academic papers, produce a structured output that extracts and expands the query into a format suitable for academic paper search APIs.

## Critical Rules

1. ALL text fields intended for search APIs must be in English (ASCII only). This includes core_concepts, methodologies, sub_queries_for_retrieval, and all fields inside api_payload_translation. NEVER output Chinese, Japanese, or Korean characters in these fields.

2. Each sub_query in sub_queries_for_retrieval MUST have a corresponding entry in BOTH semantic_scholar and openalex inside api_payload_translation — same count, same order.

3. Follow the field descriptions in the function schema for detailed guidance on what each field should contain.

## User Query

{query}
"""

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
