from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .config import load_settings
from .pipeline import run_pipeline

logger = logging.getLogger(__name__)

settings = load_settings()

app = FastAPI(
    title="Paper Search API",
    description="学术论文检索系统 — 输入自然语言查询，自动完成查询理解、多源检索、去重和结果整理",
    version="0.1.0",
)

OUTPUT_ROOT = Path(settings.output_dir)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================ API Endpoints ============================ #


@app.get("/api/search", tags=["检索"], summary="检索论文（GET）")
def api_search_get(
    q: str = Query(..., description="自然语言查询，如'2022年后关于大模型幻觉控制的论文'"),
) -> dict[str, Any]:
    return _run_pipeline(q)


@app.post("/api/search", tags=["检索"], summary="检索论文（POST）")
async def api_search_post(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        query = body.get("query", "").strip()
    else:
        form = await request.form()
        query = form.get("query", "").strip()

    if not query:
        return {"error": "查询不能为空"}, 400
    return _run_pipeline(query)


@app.get("/api/runs", tags=["实验记录"], summary="列出历史运行")
def api_list_runs() -> list[str]:
    if not OUTPUT_ROOT.exists():
        return []
    return sorted(d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir())


@app.get("/api/runs/{run_id}", tags=["实验记录"], summary="获取某次运行详情")
def api_get_run(run_id: str) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.is_dir():
        return {"error": f"运行 {run_id} 不存在"}
    result: dict[str, Any] = {"run_id": run_id}
    for name in ("result.md", "graph.json", "experiment.json"):
        fpath = run_dir / name
        if fpath.exists():
            result[name] = str(fpath)
    return result


# ============================ Web Page ============================ #


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return RedirectResponse("/search")


@app.get("/search", response_class=HTMLResponse, include_in_schema=False)
def search_page():
    return HTMLResponse(content=_SEARCH_PAGE_HTML)


_SEARCH_PAGE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Search</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #fafafa; }
  h1 { color: #1a1a2e; }
  p.desc { color: #555; margin-bottom: 24px; }
  .search-box { display: flex; gap: 8px; margin-bottom: 24px; }
  input[type="text"] { flex: 1; padding: 12px 16px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; }
  button { padding: 12px 24px; background: #1a1a2e; color: #fff; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; }
  button:hover { background: #16213e; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  .result { background: #fff; border-radius: 8px; padding: 20px; border: 1px solid #e0e0e0; }
  .result h2 { margin-top: 0; color: #1a1a2e; font-size: 18px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .stat { background: #f0f4ff; padding: 12px; border-radius: 6px; text-align: center; }
  .stat .label { font-size: 12px; color: #666; }
  .stat .value { font-size: 20px; font-weight: bold; color: #1a1a2e; }
  .paper { padding: 8px 0; border-bottom: 1px solid #f0f0f0; }
  .paper:last-child { border-bottom: none; }
  .paper .title { font-weight: 500; }
  .paper .meta { font-size: 13px; color: #888; }
  .error { color: #d32f2f; background: #fdeaea; padding: 12px; border-radius: 6px; }
  .api-link { font-size: 13px; color: #888; margin-top: 32px; }
  .api-link a { color: #1a1a2e; }
</style>
</head>
<body>
<h1>📄 Paper Search</h1>
<p class="desc">输入自然语言查询，调用 Semantic Scholar 与 OpenAlex 真实学术检索 API，自动完成查询理解、多源检索、去重和结果整理。</p>

<div class="search-box">
  <input type="text" id="query" placeholder="如：2022年后关于大模型幻觉控制、使用强化学习方法、在CVPR发表的论文" />
  <button id="btn" onclick="doSearch()">检索</button>
</div>

<div id="result"></div>

<p class="api-link">API 文档：<a href="/docs">Swagger UI</a> · <a href="/redoc">ReDoc</a></p>

<script>
async function doSearch() {
  const query = document.getElementById('query').value.trim();
  if (!query) return;
  const btn = document.getElementById('btn');
  const resultDiv = document.getElementById('result');
  btn.disabled = true;
  btn.textContent = '检索中...';
  resultDiv.innerHTML = '<p>正在调用学术 API 检索，请稍候...</p>';

  try {
    const resp = await fetch('/api/search?q=' + encodeURIComponent(query));
    const data = await resp.json();
    if (data.error) { resultDiv.innerHTML = '<p class="error">' + data.error + '</p>'; return; }

    const papers = data.papers || [];
    const eval_ = data.evaluation || {};
    const outputFiles = data.output_files || {};

    let html = '<div class="result">';
    html += '<h2>检索结果</h2>';
    html += '<div class="stats">';
    html += '<div class="stat"><div class="label">候选池</div><div class="value">' + papers.length + '</div></div>';
    html += '<div class="stat"><div class="label">Precision</div><div class="value">' + (eval_.precision ?? '-') + '</div></div>';
    html += '<div class="stat"><div class="label">Recall</div><div class="value">' + (eval_.recall ?? '-') + '</div></div>';
    html += '<div class="stat"><div class="label">F1</div><div class="value">' + (eval_.f1 ?? '-') + '</div></div>';
    html += '</div>';

    if (papers.length > 0) {
      html += '<div style="max-height:400px;overflow-y:auto">';
      papers.slice(0, 20).forEach(function(p) {
        html += '<div class="paper">';
        html += '<div class="title">' + (p.title || '(无标题)') + '</div>';
        html += '<div class="meta">' + (p.year || '') + (p.venue ? ' · ' + p.venue : '') + (p.citation_count != null ? ' · 引用' + p.citation_count : '') + ' · ' + (p.source_api || '') + '</div>';
        html += '</div>';
      });
      if (papers.length > 20) html += '<p style="color:#888;font-size:13px">仅展示前 20 篇，共 ' + papers.length + ' 篇</p>';
      html += '</div>';
    }

    if (outputFiles.markdown || outputFiles.experiment) {
      html += '<div style="margin-top:12px;font-size:13px;color:#666">';
      html += '输出文件：';
      if (outputFiles.markdown) html += ' result.md';
      if (outputFiles.graph) html += ' · graph.json';
      if (outputFiles.experiment) html += ' · experiment.json';
      html += '</div>';
    }

    html += '</div>';
    resultDiv.innerHTML = html;
  } catch (e) {
    resultDiv.innerHTML = '<p class="error">请求失败：' + e.message + '</p>';
  } finally {
    btn.disabled = false;
    btn.textContent = '检索';
  }
}

document.getElementById('query').addEventListener('keydown', function(e) { if (e.key === 'Enter') doSearch(); });
</script>
</body>
</html>"""


# ============================ Helper ============================ #


def _run_pipeline(query: str) -> dict[str, Any]:
    try:
        result = run_pipeline(query=query, backend="live", output_root=OUTPUT_ROOT, settings=settings)
        return {
            "run_id": result["run_id"],
            "papers": result["papers"],
            "evaluation": result["evaluation"],
            "output_files": result["output_files"],
        }
    except Exception as e:
        logger.exception("Pipeline failed")
        return {"error": str(e)}


def main() -> None:
    import uvicorn
    uvicorn.run("paper_search.web:app", host="127.0.0.1", port=8000, reload=False)
