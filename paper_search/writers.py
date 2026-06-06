from __future__ import annotations

import json
from pathlib import Path

from .contracts import ExperimentRecord, GraphOutput, Paper, QueryPlan


def write_outputs(
    run_dir: Path,
    query_plan: QueryPlan,
    papers: list[Paper],
    graph_output: GraphOutput,
    experiment_record: ExperimentRecord,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    result_path = run_dir / "result.md"
    graph_path = run_dir / "graph.json"
    experiment_path = run_dir / "experiment.json"

    result_path.write_text(_render_markdown(query_plan, papers), encoding="utf-8")
    graph_path.write_text(json.dumps(graph_output.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    experiment_path.write_text(json.dumps(experiment_record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "markdown": str(result_path),
        "graph": str(graph_path),
        "experiment": str(experiment_path),
        "log": str(run_dir / "logs"),
    }


def _render_markdown(query_plan: QueryPlan, papers: list[Paper]) -> str:
    high_relevance = [paper for paper in papers if paper.llm_relevance in ("高度相关", "highly_relevant")]
    partial_relevance = [paper for paper in papers if paper.llm_relevance in ("部分相关", "partially_relevant")]
    lines = [
        "# 查询摘要\n",
        f"- 原始查询：{query_plan.original_query}\n",
        f"- query_type: {query_plan.intent_analysis.query_type}\n",
        f"- 检索子查询：{', '.join(query_plan.sub_queries_for_retrieval)}\n",
        f"- 检索说明：{query_plan.intent_analysis.boundary_note or 'N/A'}\n",
        "\n# 高度相关论文列表\n",
    ]
    if high_relevance:
        _append_paper_section(lines, high_relevance)
    else:
        lines.append("- 暂无\n")

    lines.append("\n# 部分相关论文列表\n")
    if partial_relevance:
        _append_paper_section(lines, partial_relevance)
    else:
        lines.append("- 暂无\n")

    lines.extend([
        "\n# 引文关系说明\n",
        "- graph.json 中包含当前结果集的节点与边。\n",
        "\n# 运行摘要\n",
        f"- 输出论文数：{len(papers)}\n",
    ])
    return "\n".join(lines) + "\n"


def _append_paper_section(lines: list[str], papers: list[Paper]) -> None:
    for paper in papers:
        author_names = [author.get("name") or "" for author in paper.authors]
        authors = ", ".join([name for name in author_names if name]) or "Unknown"
        lines.extend(
            [
                f"## {paper.title} ({paper.year or 'Unknown'})\n",
                f"- 作者：{authors}\n",
                f"- Venue：{paper.venue or 'Unknown'}\n",
                f"- 链接：{paper.url or 'N/A'}\n",
                f"- 相关性：{paper.llm_relevance or 'highly_relevant'}\n",
                f"- 匹配理由：{paper.reason or '离线最小闭环暂未生成，当前按 query/venue/topic 命中归入结果集'}\n",
                f"- 核心贡献：{paper.contribution or '离线最小闭环暂未生成，后续由精筛阶段补充'}\n",
                f"- 摘要：{paper.abstract or 'N/A'}\n",
            ]
        )
