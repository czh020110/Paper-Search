from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

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


def _api_key_status() -> dict[str, bool]:
    return {
        "llm": bool(os.getenv("LLM_API_KEY")),
        "semantic_scholar": bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
        "openalex_mailto": bool(os.getenv("OPENALEX_MAILTO")),
        "embedding": bool(os.getenv("EMBEDDING_API_KEY")),
        "reranker": bool(os.getenv("RERANKER_API_KEY")),
    }


@app.get("/api/status", tags=["系统"], summary="API Key 配置状态")
def api_status() -> dict[str, Any]:
    return {"keys": _api_key_status(), "backend": "live"}


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
        query = str(body.get("query", "")).strip()
    else:
        form = await request.form()
        query = str(form.get("query", "")).strip()

    if not query:
        return JSONResponse({"error": "查询不能为空"}, status_code=400)  # type: ignore[return-value]
    return _run_pipeline(query)


@app.get("/api/runs", tags=["实验记录"], summary="列出历史运行")
def api_list_runs() -> list[str]:
    if not OUTPUT_ROOT.exists():
        return []
    return sorted(d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir())


@app.get("/api/runs/{run_id}/experiment", tags=["实验记录"], summary="获取实验记录JSON")
def api_get_experiment(run_id: str) -> dict[str, Any]:
    exp_path = OUTPUT_ROOT / run_id / "experiment.json"
    if exp_path.is_file():
        return json.loads(exp_path.read_text(encoding="utf-8"))
    return {"error": "experiment.json not found"}


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


@app.get("/api/search/stream", tags=["检索"], summary="流式检索（SSE）")
async def api_search_stream(q: str = Query(..., description="自然语言查询")):
    """Stream pipeline progress as Server-Sent Events."""
    import asyncio as _asyncio
    import queue as _sync_queue
    from concurrent.futures import ThreadPoolExecutor

    msg_queue: _sync_queue.Queue[dict[str, str]] = _sync_queue.Queue()

    def _on_stage(stage: str, status: str) -> None:
        try:
            msg_queue.put({"stage": stage, "status": status})
        except Exception:
            pass

    async def _stream():
        yield f"event: keys\ndata: {json.dumps(_api_key_status(), ensure_ascii=False)}\n\n"

        loop = _asyncio.get_event_loop()
        future = loop.run_in_executor(None, lambda: _run_pipeline(q, _on_stage))

        stages_seen: set[str] = set()
        while not future.done():
            try:
                msg = msg_queue.get_nowait()
                if msg["stage"] not in stages_seen:
                    stages_seen.add(msg["stage"])
                    yield f"event: progress\ndata: {json.dumps(msg, ensure_ascii=False)}\n\n"
            except _sync_queue.Empty:
                await _asyncio.sleep(0.2)

        while True:
            try:
                msg = msg_queue.get_nowait()
                if msg["stage"] not in stages_seen:
                    stages_seen.add(msg["stage"])
                    yield f"event: progress\ndata: {json.dumps(msg, ensure_ascii=False)}\n\n"
            except _sync_queue.Empty:
                break

        try:
            result = await future
            exp_path = OUTPUT_ROOT / result.get("run_id", "") / "experiment.json"
            timings_data: dict[str, float] = {}
            if exp_path.is_file():
                try:
                    exp = json.loads(exp_path.read_text(encoding="utf-8"))
                    timings_data = exp.get("stage_metrics", {}).get("overall", {}).get("stage_timings", {})
                except Exception:
                    pass
            yield f"event: result\ndata: {json.dumps({'run_id': result.get('run_id', ''), 'papers': result.get('papers', []), 'evaluation': result.get('evaluation', {}), 'output_files': result.get('output_files', {}), 'timings': timings_data}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream; charset=utf-8")


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
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; background: #fafafa; }
  h1 { color: #1a1a2e; }
  p.desc { color: #555; margin-bottom: 24px; }
  .search-box { display: flex; gap: 8px; margin-bottom: 24px; }
  input[type="text"] { flex: 1; padding: 12px 16px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; }
  button { padding: 12px 24px; background: #1a1a2e; color: #fff; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; }
  button:hover { background: #16213e; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  .result { background: #fff; border-radius: 8px; padding: 20px; border: 1px solid #e0e0e0; }
  .result h2 { margin-top: 0; color: #1a1a2e; font-size: 18px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; margin-bottom: 16px; }
  .stat { background: #f0f4ff; padding: 10px; border-radius: 6px; text-align: center; }
  .stat .label { font-size: 11px; color: #666; }
  .stat .value { font-size: 18px; font-weight: bold; color: #1a1a2e; }
  .paper { padding: 10px 0; border-bottom: 1px solid #f0f0f0; }
  .paper:hover { position: relative; z-index: 100; }
  .paper:last-child { border-bottom: none; }
  .paper .title { font-weight: 600; color: #1a1a2e; margin-bottom: 4px; }
  .paper .title a { color: inherit; text-decoration: none; }
  .paper .title a:hover { text-decoration: underline; }
  .paper .meta { font-size: 13px; color: #888; margin-bottom: 2px; }
  .paper .badges { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-top: 4px; }
  .badge { font-size: 12px; padding: 2px 8px; border-radius: 4px; font-weight: 500; white-space: nowrap; }
  .badge-high { background: #e8f5e9; color: #2e7d32; }
  .badge-partial { background: #fff3e0; color: #e65100; }
  .badge-score { background: #e3f2fd; color: #1565c0; }
  .badge-source { background: #f3e5f5; color: #7b1fa2; }
  .badge-year { background: #f5f5f5; color: #616161; }
  /* Tooltip — pops BELOW the trigger so it's never obscured by the stats bar */
  .hover-tip { position: relative; display: inline-block; }
  .hover-tip .tip-popup { display: none; position: absolute; top: 100%; left: 0; background: #1a1a2e; color: #fff; padding: 10px 14px; border-radius: 6px; font-size: 13px; width: 380px; max-height: 220px; overflow-y: auto; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.2); line-height: 1.5; }
  .tip-popup::-webkit-scrollbar { width: 6px; }
  .tip-popup::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.3); border-radius: 3px; }
  .hover-tip:hover .tip-popup, .tip-popup:hover { display: block; }
  .tip-popup .tip-label { color: #a0c4ff; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .tooltip-trigger { cursor: help; border-bottom: 1px dotted #999; display: inline-block; }
  .error { color: #d32f2f; background: #fdeaea; padding: 12px; border-radius: 6px; }
  .api-link { font-size: 13px; color: #888; margin-top: 32px; }
  .api-link a { color: #1a1a2e; }
</style>
</head>
<body>
<h1>📄 Paper Search</h1>
<p class="desc">输入自然语言查询，自动完成查询理解、多源检索、粗筛、重排序与精筛。</p>

<div id="key-status" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px"></div>

<div class="search-box">
  <input type="text" id="query" placeholder="如：2022年后关于大模型幻觉控制、使用强化学习方法、在CVPR发表的论文" />
  <button id="btn" onclick="doSearch()">检索</button>
</div>

<div id="progress" style="display:none;margin-bottom:16px">
  <div id="stage-list" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px"></div>
  <div id="progress-bar" style="height:4px;background:#e0e0e0;border-radius:2px;overflow:hidden">
    <div id="progress-fill" style="height:100%;width:0;background:#1a1a2e;transition:width 0.3s;border-radius:2px"></div>
  </div>
</div>

<div id="result"></div>

<p class="api-link">API 文档：<a href="/docs">Swagger UI</a> · <a href="/redoc">ReDoc</a></p>

<script>
const STAGES = ['query_understanding','initial_retrieval','snowball','coarse','rerank','judge'];
const STAGE_LABELS = {'query_understanding':'🔍 查询理解','initial_retrieval':'📡 多源检索','snowball':'❄️ 滚雪球','coarse':'🔢 粗筛','rerank':'🎯 重排序','judge':'⚖️ 精筛判定'};
const STAGE_SKIPPED = new Set(['snowball']);  // snowball is stubs — always skipped

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function fmtScore(s) { return s != null ? s.toFixed(2) : '-'; }
function keyBadge(label, ok) { return '<span class="badge '+(ok?'badge-high':'badge-partial')+'" style="font-size:11px">'+(ok?'✅ ':'⚠️ ')+label+'</span>'; }

// Load API key status on page load
fetch('/api/status').then(r=>r.json()).then(d=>{
  const div = document.getElementById('key-status');
  const k = d.keys;
  div.innerHTML = keyBadge('LLM',k.llm) + keyBadge('S2',k.semantic_scholar) + keyBadge('OA mailto',k.openalex_mailto) + keyBadge('Embedding',k.embedding) + keyBadge('Reranker',k.reranker);
});

// Build stage indicator HTML

function buildStages() {
  let html = '';
  STAGES.forEach(s => {
    const skipClass = STAGE_SKIPPED.has(s) ? ' style="text-decoration:line-through;opacity:0.3"' : ' style="opacity:0.4"';
    html += '<span id="sg-'+s+'" class="badge badge-year"'+skipClass+'>'+STAGE_LABELS[s]+'</span>';
  });
  return html;
}

function setStage(id, done) {
  if (STAGE_SKIPPED.has(id)) return; // skip stubs
  const el = document.getElementById('sg-'+id);
  if (el) { el.style.opacity = done ? '1' : '0.6'; }
}

function applyTimings(times) {
  if (!times || Object.keys(times).length === 0) return;
  STAGES.forEach(s => {
    const ms = times[s] || 0;
    const el = document.getElementById('sg-'+s);
    if (el && ms > 0) {
      const label = ms > 1000 ? (ms/1000).toFixed(1)+'s' : ms.toFixed(0)+'ms';
      el.innerHTML = STAGE_LABELS[s] + ' <span style="font-size:10px">' + label + '</span>';
      el.style.opacity = '1';
    }
  });
}
function setProgress(n, total) {
  document.getElementById('progress-fill').style.width = Math.round(100*n/total)+'%';
}

async function doSearch() {
  const query = document.getElementById('query').value.trim();
  if (!query) return;
  const btn = document.getElementById('btn');
  const resultDiv = document.getElementById('result');
  const progressDiv = document.getElementById('progress');
  btn.disabled = true; btn.textContent = '检索中...';
  resultDiv.innerHTML = '';

  // Show progress UI
  document.getElementById('stage-list').innerHTML = buildStages();
  progressDiv.style.display = 'block';

  let doneCount = 0;
  setProgress(0, STAGES.length);

  // Use SSE streaming endpoint
  const evtSource = new EventSource('/api/search/stream?q=' + encodeURIComponent(query));
  let papers = [], evaluation = {};

  evtSource.addEventListener('keys', function(e) {
    const k = JSON.parse(e.data);
    const div = document.getElementById('key-status');
    div.innerHTML = keyBadge('LLM',k.llm) + keyBadge('S2',k.semantic_scholar) + keyBadge('OA mailto',k.openalex_mailto) + keyBadge('Embedding',k.embedding) + keyBadge('Reranker',k.reranker);
  });

  evtSource.addEventListener('progress', function(e) {
    const d = JSON.parse(e.data);
    if (!STAGE_SKIPPED.has(d.stage)) {
      setStage(d.stage, true);
      doneCount++;
      setProgress(doneCount, STAGES.length - STAGE_SKIPPED.size);
    }
  });

  evtSource.addEventListener('result', function(e) {
    evtSource.close();
    const d = JSON.parse(e.data);
    papers = d.papers || [];
    evaluation = d.evaluation || {};

    // Render results
    let html = '<div class="result"><h2>检索结果</h2>';
    html += '<div class="stats">';
    html += '<div class="stat"><div class="label">候选池</div><div class="value">' + papers.length + '</div></div>';
    html += '<div class="stat"><div class="label">Precision</div><div class="value">' + (evaluation.precision ?? '-') + '</div></div>';
    html += '<div class="stat"><div class="label">Recall</div><div class="value">' + (evaluation.recall ?? '-') + '</div></div>';
    html += '<div class="stat"><div class="label">F1</div><div class="value">' + (evaluation.f1 ?? '-') + '</div></div>';
    html += '</div>';
    if (papers.length > 0) {
      html += '<div style="max-height:500px;overflow-y:auto">';
      const high = papers.filter(p => (p.llm_relevance || '').startsWith('高度'));
      const partial = papers.filter(p => (p.llm_relevance || '').startsWith('部分'));
      const allSorted = high.concat(partial);
      allSorted.slice(0, 50).forEach(function(p) {
        const rel = p.llm_relevance || '?';
        const score = fmtScore(p.reranker_score);
        const relBadge = rel.startsWith('高度') ? 'badge-high' : 'badge-partial';
        const authors = (p.authors || []).map(a => a.name || '').filter(Boolean).join(', ') || 'Unknown';
        const reason = p.reason || '';
        const contrib = p.contribution || '';
        const abstract = (p.abstract || '');  // keep full abstract for tooltip

        html += '<div class="paper">';
        html += '<div class="title">' + esc(p.title || '') + '</div>';
        html += '<div class="meta">';
        html += '<span class="hover-tip"><span class="tooltip-trigger">' + esc(authors.substring(0,40)) + (authors.length > 40 ? '...' : '') + '</span>';
        if (authors) { html += '<span class="tip-popup"><span class="tip-label">作者</span><br>' + esc(authors) + '</span>'; }
        html += '</span>';
        html += ' · ' + (p.year || '') + (p.venue ? ' · ' + esc(p.venue) : '') + (p.citation_count != null ? ' · 引用' + p.citation_count : '');
        html += '</div>';
        html += '<div class="badges">';
        html += '<span class="badge ' + relBadge + '">' + esc(rel) + ' / ' + score + '</span>';
        html += '<span class="badge badge-source">' + esc(p.source_api || '') + '</span>';
        if (reason) { html += '<span class="hover-tip"><span class="badge badge-year">理由▾</span><span class="tip-popup"><span class="tip-label">匹配理由</span><br>' + esc(reason) + '</span></span>'; }
        if (contrib) { html += '<span class="hover-tip"><span class="badge badge-year">贡献▾</span><span class="tip-popup"><span class="tip-label">核心贡献</span><br>' + esc(contrib) + '</span></span>'; }
        if (abstract) { html += '<span class="hover-tip"><span class="badge badge-year">摘要▾</span><span class="tip-popup"><span class="tip-label">摘要</span><br>' + esc(abstract) + '</span></span>'; }
        html += '</div></div>';
      });
      if (papers.length > 50) html += '<p style="color:#888;font-size:13px">仅展示前 50 篇，共 ' + papers.length + ' 篇</p>';
      html += '</div>';
    } else {
      html += '<p>无结果</p>';
    }
    html += '</div>';
    resultDiv.innerHTML = html;
    // Apply timings from SSE result event directly as tooltip on stage labels
    if (d.timings) { applyTimings(d.timings); }
    btn.disabled = false;
    btn.textContent = '检索';
  });

  evtSource.addEventListener('error', function(e) {
    evtSource.close();
    resultDiv.innerHTML = '<p class="error">检索失败：连接中断或超时</p>';
    progressDiv.style.display = 'none';
    btn.disabled = false;
    btn.textContent = '检索';
  });
}

document.getElementById('query').addEventListener('keydown', function(e) { if (e.key === 'Enter') doSearch(); });
</script>
</body>
</html>"""


# ============================ Helper ============================ #


def _run_pipeline(query: str, on_stage: Any = None) -> dict[str, Any]:
    try:
        result = run_pipeline(query=query, backend="live", output_root=OUTPUT_ROOT, settings=settings, on_stage=on_stage)
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
