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

PROMPT_REGISTRY: dict[str, dict] = {
    "query_understanding": {
        "name": "query_understanding",
        "version": "v1",
        "description": "Convert natural language academic query to structured QueryPlan JSON",
        "template": QUERY_UNDERSTANDING_PROMPT,
        "expected_schema": QUERY_UNDERSTANDING_SCHEMA,
    },
}
