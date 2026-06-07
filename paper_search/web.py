from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Query, Request
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


def _api_key_status() -> dict[str, Any]:
    return {
        "llm": bool(os.getenv("LLM_API_KEY")),
        "semantic_scholar": bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
        "openalex_mailto": bool(os.getenv("OPENALEX_MAILTO")),
        "embedding": bool(os.getenv("EMBEDDING_API_KEY")),
        "reranker": bool(os.getenv("RERANKER_API_KEY")),
    }


@app.get("/api/config/validate", tags=["配置"], summary="检查必需配置是否完整")
def api_validate_config() -> dict[str, Any]:
    return _is_config_ready()


@app.get("/api/status", tags=["系统"], summary="API Key 配置状态")
def api_status() -> dict[str, Any]:
    keys = _api_key_status()
    # embedding badge: grey=disabled, red=missing, green=ok
    embedding_enabled = os.getenv("EMBEDDING_ENABLED", "true").lower() not in ("0", "false", "no")
    emb_has_key = bool(os.getenv("EMBEDDING_API_KEY"))
    if not embedding_enabled:
        keys["embedding"] = "disabled"
    else:
        keys["embedding"] = emb_has_key
    return {"keys": keys, "backend": "live", "config_ready": _is_config_ready()["ready"]}


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


# ============================ Config Endpoints ============================ #


@app.get("/api/config", tags=["配置"], summary="获取当前配置")
def api_get_config() -> dict[str, Any]:
    """Return all runtime-configurable settings with their current values."""
    return _collect_config()


@app.post("/api/config", tags=["配置"], summary="更新配置（写入 .env）")
async def api_update_config(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Update one or more config values in the .env file and return the new state."""
    allowed = _config_schema()
    updated: dict[str, str] = {}
    for key, value in body.items():
        if key in allowed and value is not None:
            updated[key] = str(value)
    if updated:
        _write_env(updated)
        # Also update os.environ so the running process picks up changes immediately
        for k, v in updated.items():
            os.environ[k] = v
    return {"ok": True, "config": _collect_config()}


def _config_schema() -> dict[str, dict[str, Any]]:
    """Return the set of writable config keys with their types and defaults."""
    return {
        # Model / Provider
        "LLM_PROVIDER": {"type": "select", "default": "openai", "options": ["openai", "dashscope"]},
        "LLM_MODEL": {"type": "string", "default": "gpt-4o-mini"},
        "LLM_FAST_MODEL": {"type": "string", "default": "gpt-4o-mini"},
        "OPENAI_BASE_URL": {"type": "string", "default": "https://api.openai.com/v1"},
        "LLM_API_KEY": {"type": "string", "default": ""},
        "LLM_THINKING": {"type": "select", "default": "none", "options": ["off", "none", "minimal", "low", "medium", "high", "xhigh"]},
        "SEMANTIC_SCHOLAR_API_KEY": {"type": "string", "default": ""},
        "OPENALEX_MAILTO": {"type": "string", "default": ""},
        "EMBEDDING_PROVIDER": {"type": "select", "default": "", "options": ["", "dashscope"]},
        "EMBEDDING_MODEL": {"type": "string", "default": "text-embedding-v4"},
        "EMBEDDING_API_KEY": {"type": "string", "default": ""},
        "RERANKER_PROVIDER": {"type": "select", "default": "", "options": ["", "dashscope"]},
        "RERANKER_MODEL": {"type": "string", "default": "qwen3-rerank"},
        "RERANKER_API_KEY": {"type": "string", "default": ""},
        # Coarse screening
        "COARSE_POOL_SKIP_THRESHOLD": {"type": "int", "default": "40"},
        "COARSE_EMBEDDING_MIN_SIMILARITY": {"type": "float", "default": "0.35"},
        "COARSE_WEIGHT_BM25": {"type": "float", "default": "0.35"},
        "COARSE_WEIGHT_EMBEDDING": {"type": "float", "default": "0.35"},
        "COARSE_WEIGHT_STRUCTURE": {"type": "float", "default": "0.30"},
        "COARSE_WEIGHT_BM25_FALLBACK": {"type": "float", "default": "0.60"},
        "COARSE_WEIGHT_STRUCTURE_FALLBACK": {"type": "float", "default": "0.40"},
        "COARSE_RELATIVE_THRESHOLD_FACTOR": {"type": "float", "default": "0.3"},
        # Reranker
        "RERANKER_TOP_K_FALLBACK": {"type": "int", "default": "30"},
        "RERANKER_RELATIVE_THRESHOLD_FACTOR": {"type": "float", "default": "0.7"},
        # Embedding
        "EMBEDDING_CONCURRENCY": {"type": "int", "default": "10"},
        "EMBEDDING_RPS_LIMIT": {"type": "int", "default": "15"},
        "EMBEDDING_ENABLED": {"type": "bool", "default": "true"},
        # Judge
        "JUDGE_CONCURRENCY": {"type": "int", "default": "100"},
        "JUDGE_WAVE_CAP": {"type": "int", "default": "5000"},
    }


_REQUIRED_CONFIG_KEYS = [
    "LLM_MODEL", "LLM_FAST_MODEL", "LLM_API_KEY",
    "RERANKER_API_KEY", "RERANKER_MODEL",
    "EMBEDDING_API_KEY", "EMBEDDING_MODEL",
]


def _is_config_ready() -> dict[str, Any]:
    """Check whether all required config values are set.

    Returns a dict with ``ready`` (bool) and ``missing`` (list[str]).
    ``EMBEDDING_*`` are only required when ``EMBEDDING_ENABLED`` is true.
    """
    missing: list[str] = []
    for key in _REQUIRED_CONFIG_KEYS:
        val = os.getenv(key, "").strip()
        if not val:
            missing.append(key)
    # Embedding is only required if enabled
    embedding_enabled = os.getenv("EMBEDDING_ENABLED", "true").lower() not in ("0", "false", "no")
    if not embedding_enabled:
        missing = [m for m in missing if not m.startswith("EMBEDDING_")]
    ready = len(missing) == 0
    return {"ready": ready, "missing": missing}


_CONFIG_HELP: dict[str, str] = {
    "LLM_PROVIDER": "LLM 提供商。OpenAI(兼容) 使用 reasoning_effort 控制思考；DashScope 使用 enable_thinking。示例：openai",
    "LLM_MODEL": "LLM 模型名称，用于查询理解和精筛。示例：qwen3.6-plus / gpt-4o-mini",
    "LLM_FAST_MODEL": "轻量 LLM 模型，用于快速任务。不配置则回退到 LLM_MODEL。示例：qwen3.6-flash-nothinking",
    "OPENAI_BASE_URL": "OpenAI 兼容 API 端点 URL。留空默认使用 OpenAI 官方节点 (https://api.openai.com/v1)。示例：https://api.openai.com/v1",
    "LLM_API_KEY": "LLM API 认证密钥。示例：sk-...",
    "LLM_THINKING": "控制 LLM 思考/推理深度。none=关闭，off=不传参(API默认)，minimal~xhigh=递增推理强度。DashScope 仅支持开/关。",
    "SEMANTIC_SCHOLAR_API_KEY": "S2 API Key（可选）。有 key 可提升速率限制，留空使用公共端点。示例：40字符字符串",
    "OPENALEX_MAILTO": "OpenAlex 礼貌邮箱（可选）。填入邮箱可进入礼貌池获得更稳定服务。示例：your-email@example.com",
    "EMBEDDING_PROVIDER": "Embedding 服务提供商。目前支持 dashscope。留空则跳过路 B 打分。示例：dashscope",
    "EMBEDDING_MODEL": "Embedding 模型名称。示例：text-embedding-v4",
    "EMBEDDING_API_KEY": "Embedding API 密钥。示例：sk-...",
    "RERANKER_PROVIDER": "Reranker 服务提供商。留空则跳过重排序。示例：dashscope",
    "RERANKER_MODEL": "Reranker 模型名称。示例：qwen3-rerank",
    "RERANKER_API_KEY": "Reranker API 密钥。示例：sk-...",
    "COARSE_POOL_SKIP_THRESHOLD": "候选池 ≤N 篇时跳过粗筛直接透传。示例：40",
    "COARSE_EMBEDDING_MIN_SIMILARITY": "余弦相似度硬底线。低于此值的论文直接排除。范围 0~1。示例：0.35",
    "COARSE_WEIGHT_BM25": "三路融合中 BM25 的权重。有 Embedding 时使用。示例：0.35",
    "COARSE_WEIGHT_EMBEDDING": "三路融合中 Embedding 的权重。示例：0.35",
    "COARSE_WEIGHT_STRUCTURE": "三路融合中 Structure 的权重。示例：0.30",
    "COARSE_WEIGHT_BM25_FALLBACK": "无 Embedding 时 BM25 的降级权重。示例：0.60",
    "COARSE_WEIGHT_STRUCTURE_FALLBACK": "无 Embedding 时 Structure 的降级权重。示例：0.40",
    "COARSE_RELATIVE_THRESHOLD_FACTOR": "综合分截断因子。threshold = mean × factor。示例：0.3",
    "RERANKER_TOP_K_FALLBACK": "重排序 Top-K 保底数量。至少保留这么多篇。示例：30",
    "RERANKER_RELATIVE_THRESHOLD_FACTOR": "重排序相对阈值因子。threshold = mean × factor。示例：0.7",
    "EMBEDDING_CONCURRENCY": "Embedding 并发 worker 数。过大触发 API 限流。示例：10",
    "EMBEDDING_RPS_LIMIT": "Embedding API 每秒最大请求数。token bucket 限流。示例：15",
    "EMBEDDING_ENABLED": "是否启用 Embedding 稠密检索。关闭后三路退化为 BM25+Structure。",
    "JUDGE_CONCURRENCY": "精筛 LLM 并发调用数。示例：100",
    "JUDGE_WAVE_CAP": "精筛单次 wave 处理上限。超过此数量分批处理。示例：5000",
}


def _collect_config() -> dict[str, Any]:
    """Gather current config values from env, falling back to defaults.

    Returns a dict where each value is either the raw env value or an object
    like ``{"value": "...", "options": [...]}`` for select-type keys so the
    frontend knows how to render them.  Includes ``help`` text and ``required``
    flag for each key.
    """
    schema = _config_schema()
    required = set(_REQUIRED_CONFIG_KEYS)
    validate = _is_config_ready()
    result: dict[str, Any] = {}
    for key, spec in schema.items():
        raw = os.getenv(key)
        val = raw if raw is not None else spec["default"]
        entry: dict[str, Any] = {
            "value": val,
            "help": _CONFIG_HELP.get(key, ""),
            "required": key in required,
        }
        if spec["type"] == "select":
            entry["options"] = spec.get("options", [])
        result[key] = entry
    result["_validate"] = validate
    return result


def _env_path() -> Path:
    """Return .env.local path (local-only, not committed); fallback to .env."""
    local = Path.cwd() / ".env.local"
    if local.is_file():
        return local
    return Path.cwd() / ".env"


def _write_env(updates: dict[str, str]) -> None:
    """Write config updates to .env.local, creating from .env template if needed.

    Keys that already exist are updated in-place; new keys are appended.
    Comment lines and blank lines are preserved.
    """
    local = Path.cwd() / ".env.local"
    env = Path.cwd() / ".env"

    # Auto-create .env.local from .env template if it doesn't exist
    if not local.is_file() and env.is_file():
        local.write_text(env.read_text(encoding="utf-8"), encoding="utf-8")

    env_file = local if local.is_file() else env
    lines: list[str] = []
    seen: set[str] = set()
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines(keepends=True):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                if "=" in stripped:
                    key = stripped.split("=", 1)[0].strip()
                    if key in updates:
                        lines.append(f'{key}={updates[key]}\n')
                        seen.add(key)
                        continue
            lines.append(line if line.endswith("\n") else line + "\n")
    # Append keys not yet seen
    for key, value in updates.items():
        if key not in seen:
            lines.append(f"{key}={value}\n")
    env_file.write_text("".join(lines), encoding="utf-8")


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
  button.btn-interrupt { background: #d32f2f; }
  button.btn-interrupt:hover { background: #b71c1c; }
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
  .hover-tip .tip-popup { display: none; position: absolute; top: 100%; left: 0; background: #1a1a2e; color: #fff; padding: 10px 14px; border-radius: 6px; font-size: 13px; width: 380px; max-height: 220px; overflow-y: auto; overflow-x: hidden; word-break: break-all; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.2); line-height: 1.5; }
  .tip-popup::-webkit-scrollbar { width: 4px; }
  .tip-popup::-webkit-scrollbar-track { background: transparent; }
  .tip-popup::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.25); border-radius: 2px; }
  .hover-tip:hover .tip-popup, .tip-popup:hover { display: block; }
  .tip-popup .tip-label { color: #a0c4ff; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .tooltip-trigger { cursor: help; border-bottom: 1px dotted #999; display: inline-block; }
  .error { color: #d32f2f; background: #fdeaea; padding: 12px; border-radius: 6px; }
  .api-link { font-size: 13px; color: #888; margin-top: 32px; }
  .api-link a { color: #1a1a2e; }
  /* Gear button — fixed top-right */
  .gear-btn { position: fixed; top: 16px; right: 16px; width: 40px; height: 40px; border-radius: 8px; background: #9e9e9e; color: #fff; border: none; font-size: 20px; cursor: pointer; display: flex; align-items: center; justify-content: center; z-index: 200; }
  .gear-btn:hover { background: #757575; }
  /* Modal overlay */
  .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.4); z-index: 300; align-items: center; justify-content: center; }
  .modal-overlay.show { display: flex; }
  .modal { background: #fff; border-radius: 12px; padding: 24px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; box-shadow: 0 8px 32px rgba(0,0,0,0.2); }
  .modal::-webkit-scrollbar { width: 4px; }
  .modal::-webkit-scrollbar-thumb { background: #ccc; border-radius: 2px; }
  .modal h2 { margin-top: 0; color: #1a1a2e; }
  .modal h3 { color: #1a1a2e; font-size: 14px; margin: 16px 0 8px; padding-top: 12px; border-top: 1px solid #eee; }
  .modal h3:first-of-type { border-top: none; padding-top: 0; }
  .modal .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 6px; }
  .modal label { font-size: 12px; color: #666; display: block; margin-bottom: 2px; }
  .modal input, .modal select { width: 100%; padding: 6px 8px; font-size: 13px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
  .modal input[type="checkbox"] { width: auto; margin-left: 4px; }
  .modal .actions { display: flex; gap: 8px; margin-top: 16px; justify-content: flex-end; }
  .modal .actions button { padding: 8px 20px; font-size: 14px; }
  .modal .btn-secondary { background: #e0e0e0; color: #333; }
  .modal .btn-secondary:hover { background: #ccc; }
  .modal .toast { display: none; position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); background: #2e7d32; color: #fff; padding: 10px 24px; border-radius: 8px; font-size: 14px; z-index: 400; }
  .modal .toast.show { display: block; }
  /* Help icon beside config field labels */
  .help-icon { display: inline-block; margin-left: 4px; width: 16px; height: 16px; border-radius: 50%; background: #e0e0e0; color: #666; font-size: 10px; font-weight: bold; cursor: default; text-align: center; line-height: 16px; vertical-align: top; position: relative; }
  .help-icon:hover { background: #bdbdbd; }
  .help-icon-missing { background: #fdeaea; color: #d32f2f; }
  .help-icon-missing:hover { background: #f5c6c6; }
  .help-icon-ok { background: #e8f5e9; color: #2e7d32; }
  .help-icon-ok:hover { background: #c8e6c9; }
  /* Help tooltip — shows below the icon */
  .help-icon .help-tip { display: none; position: absolute; top: 100%; left: 0; background: #1a1a2e; color: #fff; padding: 6px 10px; border-radius: 6px; font-size: 12px; width: 260px; max-height: 160px; overflow-y: auto; z-index: 500; box-shadow: 0 4px 12px rgba(0,0,0,0.2); line-height: 1.4; font-weight: normal; white-space: normal; text-align: left; margin-top: 4px; }
  .help-icon .help-tip::-webkit-scrollbar { width: 4px; }
  .help-icon .help-tip::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.25); border-radius: 2px; }
  .help-icon:hover .help-tip, .help-tip:hover { display: block; }
  /* Test buttons */
  .test-btn { font-size: 11px; padding: 3px 10px; border: none; border-radius: 4px; cursor: pointer; margin-left: 6px; vertical-align: middle; }
  .test-btn-ok { background: #e8f5e9; color: #2e7d32; }
  .test-btn-ok:hover { background: #c8e6c9; }
  .test-btn-fail { background: #fdeaea; color: #d32f2f; }
  .test-btn-fail:hover { background: #f5c6c6; }
  .test-btn-idle { background: #e3f2fd; color: #1565c0; }
  .test-btn-idle:hover { background: #bbdefb; }
  .test-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  /* Embedding disabled state */
  .emb-disabled { opacity: 0.4; pointer-events: none; }
</style>
</head>
<body>
<button class="gear-btn" onclick="toggleConfig()" title="配置">⚙</button>
<h1>📄 Paper Search</h1>
<p class="desc">输入自然语言查询，自动完成查询理解、多源检索、粗筛、重排序与精筛。<span class="hover-tip" style="display:inline-block;vertical-align:middle"><span class="tooltip-trigger" style="font-size:14px;cursor:default">❓</span><span class="tip-popup" style="width:520px;max-height:420px;font-size:12px;line-height:1.6">
<b>🔍 查询理解</b><br>
调用 <b>LLM_FAST_MODEL</b> 将自然语言转为结构化 QueryPlan JSON。<br>
使用 <b>with_structured_output</b> + Pydantic 强制输出 schema。<br>
思考程度由 <b>LLM_THINKING</b> 控制。<br>
API: <b>LLM_PROVIDER</b> → <b>OPENAI_BASE_URL</b><br>
输出: query_type, hard_filters, ranking_signals, semantic_queries, sub_queries_for_retrieval, api_payload_translation<br><br>

<b>📡 多源检索</b><br>
对 sub_queries_for_retrieval 逐条调用 <b>Semantic Scholar</b> + <b>OpenAlex</b> API。<br>
可选参数: <b>SEMANTIC_SCHOLAR_API_KEY</b>（有 key 提升限速），<b>OPENALEX_MAILTO</b>（礼貌池）<br>
navigational 查询 → 标题精确匹配；semantic → 多角度语义扩展；metadata → 优先消费 api_payload_translation<br><br>

<b>❄️ 滚雪球</b><br>
基于 LLM 判定的高分论文自动扩展检索范围。<br>
- query_evolution: 调用 <b>LLM_FAST_MODEL</b> 从"高度相关"论文提取 3-5 个新关键词<br>
- re_search: 新关键词再次调 S2/OA API 补召回，去重后加入候选池<br>
- expand_citations: 对高分论文调 S2/OA 引文/参考文献 API，记录引文关系<br>
收敛条件: 重叠率 &gt;70%、无新关键词、最大轮数 3 或预算耗尽<br><br>

<b>🔢 粗筛</b>（候选池 &gt; <b>COARSE_POOL_SKIP_THRESHOLD</b>=40 时触发）<br>
三路混合打分（min-max 归一化 + 加权融合）:<br>
路A BM25: rank-bm25 关键词匹配<br>
路B Embedding: <b>EMBEDDING_PROVIDER</b>(<b>EMBEDDING_MODEL</b>) 余弦相似度<br>
路C Structure: 年份接近度 + log1p(引用数) + venue 模糊命中<br>
权重: BM25=<b>COARSE_WEIGHT_BM25</b> Embedding=<b>COARSE_WEIGHT_EMBEDDING</b> Structure=<b>COARSE_WEIGHT_STRUCTURE</b><br>
降级权重: BM25=<b>COARSE_WEIGHT_BM25_FALLBACK</b> Structure=<b>COARSE_WEIGHT_STRUCTURE_FALLBACK</b><br>
硬底线: cosine_sim &lt; <b>COARSE_EMBEDDING_MIN_SIMILARITY</b>=0.35 → 直接排除<br>
截断: combined &lt; mean × <b>COARSE_RELATIVE_THRESHOLD_FACTOR</b>=0.3 → 排除<br>
Embedding 并发: <b>EMBEDDING_CONCURRENCY</b>=10 workers, RPS 限速: <b>EMBEDDING_RPS_LIMIT</b>=15<br>
开关: <b>EMBEDDING_ENABLED</b><br><br>

<b>🎯 重排序</b><br>
调用 <b>RERANKER_PROVIDER</b>(<b>RERANKER_MODEL</b>) DashScope qwen3-rerank API。<br>
保底 Top-<b>RERANKER_TOP_K_FALLBACK</b>=30 篇 + 相对阈值 mean × <b>RERANKER_RELATIVE_THRESHOLD_FACTOR</b>=0.7。<br>
无 RERANKER_PROVIDER 时 pass-through 全量透传。<br><br>

<b>⚖️ 精筛</b><br>
<b>JUDGE_CONCURRENCY</b>=100 并发逐篇调用 <b>LLM_FAST_MODEL</b> 三分类判定。<br>
每篇 1 次 LLM 调用 → 高度相关/部分相关/不相关 + 理由 + 贡献。<br>
单波上限 <b>JUDGE_WAVE_CAP</b>=5000 篇。<br>
无 <b>LLM_API_KEY</b> 时报错，不静默回退。</span></span></p>

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
const STAGES = ['query_understanding','initial_retrieval','coarse','rerank','judge','snowball'];
const STAGE_LABELS = {'query_understanding':'🔍 查询理解','initial_retrieval':'📡 多源检索','snowball':'❄️ 滚雪球','coarse':'🔢 粗筛','rerank':'🎯 重排序','judge':'⚖️ 精筛判定'};
const STAGE_SKIPPED = new Set([]);  // snowball implemented (S-005), no longer skipped

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function fmtScore(s) { return s != null ? s.toFixed(2) : '-'; }
function keyBadge(label, ok) {
  var cls = ok === true ? 'badge-high' : ok === 'disabled' ? 'badge-year' : 'badge-partial';
  var icon = ok === true ? '✅ ' : ok === 'disabled' ? '⚫ ' : '⚠️ ';
  return '<span class=\"badge '+cls+'\" style=\"font-size:11px\">'+icon+label+'</span>';
}

// Load API key status on page load
refreshKeyStatus();

async function refreshKeyStatus() {
  const r = await fetch('/api/status');
  const d = await r.json();
  const div = document.getElementById('key-status');
  const k = d.keys;
  div.innerHTML = keyBadge('LLM',k.llm) + keyBadge('S2',k.semantic_scholar) + keyBadge('OA mailto',k.openalex_mailto) + keyBadge('Embedding',k.embedding) + keyBadge('Reranker',k.reranker);
}

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
  if (STAGE_SKIPPED.has(id)) return;
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

  // Cancel existing search if running
  if (btn.dataset.searching === '1') {
    if (window._evtSource) { window._evtSource.close(); window._evtSource = null; }
    btn.dataset.searching = '0';
    btn.className = '';
    btn.textContent = '检索';
    progressDiv.style.display = 'none';
    return;
  }

  // Immediately switch to interrupt state
  btn.dataset.searching = '1';
  btn.className = 'btn-interrupt';
  btn.textContent = '中断';
  resultDiv.innerHTML = '';

  // Pre-flight: check config readiness
  const vRes = await fetch('/api/config/validate');
  const vData = await vRes.json();
  if (!vData.ready) {
    btn.dataset.searching = '0';
    btn.className = '';
    btn.textContent = '检索';
    resultDiv.innerHTML = '<p class=\"error\">配置未完成，缺少必需字段：' + vData.missing.join(', ') + '<br>请点击右上角 ⚙ 完成配置后再检索。</p>';
    return;
  }

  // Show progress UI
  document.getElementById('stage-list').innerHTML = buildStages();
  progressDiv.style.display = 'block';

  let doneCount = 0;
  setProgress(0, STAGES.length);

  // Use SSE streaming endpoint
  window._evtSource = new EventSource('/api/search/stream?q=' + encodeURIComponent(query));
  const evtSource = window._evtSource;
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
    if (window._evtSource === evtSource) window._evtSource = null;
    btn.dataset.searching = '0';
    btn.className = '';
    btn.textContent = '检索';

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
        if (p.url) { html += '<span class="hover-tip"><a href="' + esc(p.url) + '" target="_blank" rel="noopener" class="badge" style="background:#e3f2fd;color:#1565c0;text-decoration:none">原文🔗</a><span class="tip-popup"><span class="tip-label">原文链接</span><br>' + esc(p.url) + '</span></span>'; }
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
    btn.dataset.searching = '0';
    btn.className = '';
    btn.textContent = '检索';
  });

  evtSource.addEventListener('error', function(e) {
    evtSource.close();
    if (window._evtSource === evtSource) window._evtSource = null;
    resultDiv.innerHTML = '<p class="error">检索失败：连接中断或超时</p>';
    progressDiv.style.display = 'none';
    btn.dataset.searching = '0';
    btn.className = '';
    btn.textContent = '检索';
  });
}

document.getElementById('query').addEventListener('keydown', function(e) { if (e.key === 'Enter') doSearch(); });

// ===================== Config Panel =====================

let configData = {};
let configDirty = false;

async function loadConfig() {
  const r = await fetch('/api/config');
  configData = await r.json();
  updateThinkingOptions();
  _renderHelpIcons();
}

function _formElements() {
  return document.querySelectorAll('#config-form input, #config-form select');
}

function _snapshotForm() {
  // Record current form values for dirty checking
  for (const el of _formElements()) {
    if (!el.id || !el.id.startsWith('cfg-')) continue;
    el.dataset.savedValue = el.type === 'checkbox' ? String(el.checked) : el.value;
  }
  configDirty = false;
}

function _isDirty() {
  for (const el of _formElements()) {
    if (!el.id || !el.id.startsWith('cfg-')) continue;
    const saved = el.dataset.savedValue || '';
    const cur = el.type === 'checkbox' ? String(el.checked) : el.value;
    if (cur !== saved) return true;
  }
  return false;
}

function _markDirty() { configDirty = true; }

function openConfig() {
  const ov = document.getElementById('config-overlay');
  // Bind dirty listener on first open (form exists by now)
  const form = document.getElementById('config-form');
  if (form && !form.dataset.listenerBound) {
    form.addEventListener('change', _markDirty);
    form.addEventListener('input', _markDirty);
    form.dataset.listenerBound = '1';
  }
  // Populate form from configData — values are now objects {value, help, required, ...}
  for (const [k, v] of Object.entries(configData)) {
    if (k === '_validate') continue;
    const el = document.getElementById('cfg-' + k);
    if (!el) continue;
    const realVal = (typeof v === 'object' && v !== null) ? (v.value || '') : (v || '');
    if (el.type === 'checkbox') el.checked = (realVal === 'true' || realVal === true);
    else el.value = realVal;
  }
  ov.classList.add('show');
  updateThinkingOptions();
  _renderHelpIcons();
  updateEmbDependent();
  _snapshotForm();
}

function closeConfig(force) {
  if (force !== true && _isDirty()) {
    if (!confirm('有未保存的修改，是否放弃？')) return;
  }
  document.getElementById('config-overlay').classList.remove('show');
}

function toggleConfig() {
  const ov = document.getElementById('config-overlay');
  if (ov.classList.contains('show')) { closeConfig(false); return; }
  openConfig();
}

async function saveConfig() {
  const body = {};
  for (const el of _formElements()) {
    if (!el.id || !el.id.startsWith('cfg-')) continue;
    const key = el.id.slice(4);
    const cv = configData[key];
    const oldVal = (typeof cv === 'object' && cv !== null) ? (cv.value || '') : String(cv || '');
    const newVal = el.type === 'checkbox' ? (el.checked ? 'true' : 'false') : el.value;
    // Allow empty string as intentional delete (only send if changed from oldVal)
    if (newVal !== String(oldVal)) body[key] = newVal;
  }
  if (Object.keys(body).length === 0) { closeConfig(true); return; }
  await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  await loadConfig();
  closeConfig(true);
  refreshKeyStatus();
  const toast = document.getElementById('config-toast');
  toast.classList.add('show');
  setTimeout(function() { toast.classList.remove('show'); }, 2000);
}

// ===================== Test Functions =====================

function showTestToast(message, isOk) {
  let toast = document.getElementById('test-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'test-toast';
    toast.className = 'toast';
    toast.style.cssText = 'display:none;position:fixed;bottom:60px;left:50%;transform:translateX(-50%);color:#fff;padding:10px 24px;border-radius:8px;font-size:14px;z-index:400;max-width:80vw;word-break:break-word';
    document.body.appendChild(toast);
  }
  toast.style.background = isOk ? '#2e7d32' : '#d32f2f';
  toast.textContent = (isOk ? '✅ ' : '❌ ') + message;
  toast.classList.add('show');
  toast.style.display = 'block';
  setTimeout(function() { toast.style.display = 'none'; toast.classList.remove('show'); }, 4000);
}

function _getFormValue(key) {
  const el = document.getElementById('cfg-' + key);
  if (!el) return '';
  return el.type === 'checkbox' ? (el.checked ? 'true' : 'false') : el.value;
}

async function _callTest(endpoint, btnId, body) {
  const btn = document.getElementById(btnId);
  if (btn) { btn.disabled = true; btn.textContent = '测试中...'; }
  try {
    const opts = { method: 'POST' };
    if (body) { opts.headers = {'Content-Type': 'application/json'}; opts.body = JSON.stringify(body); }
    const r = await fetch(endpoint, opts);
    const d = await r.json();
    showTestToast(d.message, d.ok);
    if (btn) {
      btn.className = 'test-btn ' + (d.ok ? 'test-btn-ok' : 'test-btn-fail');
      btn.textContent = d.ok ? '✓ 正常' : '✗ 失败';
      setTimeout(function() {
        btn.className = 'test-btn test-btn-idle';
        btn.textContent = '测试';
        btn.disabled = false;
      }, 4000);
    }
  } catch (e) {
    showTestToast('请求失败: ' + e.message, false);
    if (btn) { btn.disabled = false; btn.textContent = '测试'; }
  }
}

function testLLM() {
  _callTest('/api/test/llm', 'btn-test-llm', {
    provider: _getFormValue('LLM_PROVIDER'),
    api_key: _getFormValue('LLM_API_KEY'),
    model: _getFormValue('LLM_MODEL'),
    base_url: _getFormValue('OPENAI_BASE_URL'),
  });
}

function testS2() {
  _callTest('/api/test/semantic-scholar', 'btn-test-s2', {
    api_key: _getFormValue('SEMANTIC_SCHOLAR_API_KEY'),
  });
}

function testOA() {
  _callTest('/api/test/openalex', 'btn-test-oa', {
    mailto: _getFormValue('OPENALEX_MAILTO'),
  });
}

function testEmbedding() {
  const btn = document.getElementById('btn-test-emb');
  if (btn) { btn.disabled = true; btn.textContent = '测试中...'; }
  const embBody = {
    api_key: _getFormValue('EMBEDDING_API_KEY'),
    model: _getFormValue('EMBEDDING_MODEL'),
  };
  const rerankBody = {
    api_key: _getFormValue('RERANKER_API_KEY'),
    model: _getFormValue('RERANKER_MODEL'),
  };
  Promise.all([
    fetch('/api/test/embedding', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(embBody) }).then(function(r) { return r.json(); }),
    fetch('/api/test/reranker', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(rerankBody) }).then(function(r) { return r.json(); }),
  ]).then(function(d) {
    var d1 = d[0], d2 = d[1];
    var msg = 'Embedding: ' + (d1.ok ? 'OK' : 'Fail') + ' | Reranker: ' + (d2.ok ? 'OK' : 'Fail');
    var allOk = d1.ok && d2.ok;
    showTestToast(msg, allOk);
    if (btn) {
      btn.className = 'test-btn ' + (allOk ? 'test-btn-ok' : 'test-btn-fail');
      btn.textContent = allOk ? 'OK' : '部分失败';
      setTimeout(function() { btn.className = 'test-btn test-btn-idle'; btn.textContent = '测试'; btn.disabled = false; }, 4000);
    }
  }).catch(function(e) {
    showTestToast('请求失败: ' + e.message, false);
    if (btn) { btn.disabled = false; btn.textContent = '测试'; }
  });
}

// Embedding toggle: grey-out / enable dependent fields
function updateEmbDependent() {
  const cb = document.getElementById('cfg-EMBEDDING_ENABLED');
  const enabled = cb && cb.checked;
  const els = document.querySelectorAll('.emb-dependent');
  els.forEach(function(el) {
    if (enabled) {
      el.classList.remove('emb-disabled');
      el.querySelectorAll('input, select').forEach(function(inp) { inp.disabled = false; });
    } else {
      el.classList.add('emb-disabled');
      el.querySelectorAll('input, select').forEach(function(inp) { inp.disabled = true; });
    }
  });
  // Also update help icons
  _renderHelpIcons();
}

function onEmbeddingToggle() {
  updateEmbDependent();
}

// ===================== End Test Functions =====================

// Dynamic thinking dropdown: DashScope only has on/off, OpenAI has full reasoning levels
const thinkingOptions = {
  openai: [
    {value: 'off', label: '默认 (不传参数)'},
    {value: 'none', label: '关闭'},
    {value: 'minimal', label: '最低'},
    {value: 'low', label: '低'},
    {value: 'medium', label: '中'},
    {value: 'high', label: '高'},
    {value: 'xhigh', label: '极高'},
  ],
  dashscope: [
    {value: 'off', label: '默认 (不传参数)'},
    {value: 'none', label: '关闭'},
    {value: 'low', label: '开'},
  ]
};

function updateThinkingOptions() {
  const providerEl = document.getElementById('cfg-LLM_PROVIDER');
  const thinkingSel = document.getElementById('cfg-LLM_THINKING');
  if (!providerEl || !thinkingSel) return;
  const provider = providerEl.value;
  const opts = thinkingOptions[provider] || thinkingOptions.openai;
  const cv = configData['LLM_THINKING'];
  const curVal = (typeof cv === 'object' && cv !== null) ? (cv.value || 'none') : (cv || 'none');
  thinkingSel.innerHTML = '';
  opts.forEach(function(o) {
    const el = document.createElement('option');
    el.value = o.value;
    el.textContent = o.label;
    if (o.value === curVal) el.selected = true;
    thinkingSel.appendChild(el);
  });
}

// Render ❓ help icons next to each config field's label text
function _renderHelpIcons() {
  var embeddingEnabled = document.getElementById('cfg-EMBEDDING_ENABLED');
  var embDisabled = embeddingEnabled && !embeddingEnabled.checked;
  var allInputs = document.querySelectorAll('#config-form input[id], #config-form select[id]');
  allInputs.forEach(function(input) {
    if (!input.id || !input.id.startsWith('cfg-')) return;
    var key = input.id.slice(4);
    var cv = configData[key];
    if (!cv || typeof cv !== 'object') return;
    var label = input.closest('.row > div') ? input.closest('.row > div').querySelector('label') : null;
    if (!label) return;
    // Remove any existing icon for this field
    var existing = label.querySelector('.help-icon');
    if (existing) existing.remove();
    // Create new icon with tooltip
    var icon = document.createElement('span');
    icon.className = 'help-icon';
    icon.textContent = '?';
    var tip = document.createElement('span');
    tip.className = 'help-tip';
    tip.textContent = cv.help || '';
    icon.appendChild(tip);
    var val = cv.value || '';
    var isMissing = cv.required && !val;
    if (cv.required && key.startsWith('EMBEDDING_') && embDisabled) {
      isMissing = false;
    }
    if (isMissing) {
      icon.classList.add('help-icon-missing');
    } else if (cv.required) {
      icon.classList.add('help-icon-ok');
    }
    label.appendChild(icon);
  });
}

loadConfig();
</script>

<div id="config-overlay" class="modal-overlay" onclick="if(event.target===this)closeConfig(false)">
<div class="modal">
<h2>⚙ 配置</h2>
<form id="config-form" onsubmit="event.preventDefault();saveConfig()">

<h3>模型 & 供应商</h3>
<div class="row">
  <div><label>LLM Provider</label><select id="cfg-LLM_PROVIDER" onchange="updateThinkingOptions()"><option value="openai">OpenAI (兼容)</option><option value="dashscope">DashScope</option></select></div>
  <div><label>思考程度</label><select id="cfg-LLM_THINKING"></select></div>
</div>
<div class="row">
  <div><label>LLM Model</label><input id="cfg-LLM_MODEL" placeholder="gpt-4o-mini"></div>
  <div><label>LLM Fast Model</label><input id="cfg-LLM_FAST_MODEL" placeholder="gpt-4o-mini"></div>
</div>
<div class="row">
  <div><label>OpenAI Base URL</label><input id="cfg-OPENAI_BASE_URL" placeholder="https://api.openai.com/v1"></div>
  <div><label>LLM API Key</label><input id="cfg-LLM_API_KEY" type="password" placeholder="sk-..."></div>
</div>
<div style="margin:-8px 0 12px;text-align:right"><button id="btn-test-llm" type="button" class="test-btn test-btn-idle" onclick="testLLM()">🔗 测试连接</button></div>

<h3>学术搜索 API</h3>
<div class="row">
  <div><label>S2 API Key (可选)</label><input id="cfg-SEMANTIC_SCHOLAR_API_KEY" placeholder="留空使用公共端点"><button id="btn-test-s2" type="button" class="test-btn test-btn-idle" onclick="testS2()" style="margin-left:4px">测试 S2</button></div>
  <div><label>OpenAlex 礼貌邮箱 (可选)</label><input id="cfg-OPENALEX_MAILTO" placeholder="your-email@example.com"><button id="btn-test-oa" type="button" class="test-btn test-btn-idle" onclick="testOA()" style="margin-left:4px">测试 OA</button></div>
</div>

<h3>Embedding / Reranker</h3>
<div class="row">
  <div class="emb-dependent"><label>Embedding Provider</label><select id="cfg-EMBEDDING_PROVIDER"><option value="">(无)</option><option value="dashscope">DashScope</option></select></div>
  <div class="emb-dependent"><label>Embedding Model</label><input id="cfg-EMBEDDING_MODEL" placeholder="text-embedding-v4"></div>
</div>
<div class="row">
  <div class="emb-dependent"><label>Embedding API Key</label><input id="cfg-EMBEDDING_API_KEY" type="password" placeholder="sk-..."></div>
  <div class="emb-dependent"><label>Reranker Provider</label><select id="cfg-RERANKER_PROVIDER"><option value="">(无)</option><option value="dashscope">DashScope</option></select></div>
</div>
<div class="row">
  <div class="emb-dependent"><label>Reranker Model</label><input id="cfg-RERANKER_MODEL" placeholder="qwen3-rerank"></div>
  <div class="emb-dependent"><label>Reranker API Key</label><input id="cfg-RERANKER_API_KEY" type="password" placeholder="sk-..."></div>
</div>
<div class="row">
  <div class="emb-dependent"><label>并发数</label><input id="cfg-EMBEDDING_CONCURRENCY" type="number" step="1"></div>
  <div class="emb-dependent"><label>RPS 限速</label><input id="cfg-EMBEDDING_RPS_LIMIT" type="number" step="1"></div>
  <div><label>启用 Embedding</label><input id="cfg-EMBEDDING_ENABLED" type="checkbox" style="width:auto;margin:6px 0 0" onchange="onEmbeddingToggle()"></div>
</div>
<div style="margin:-8px 0 12px;text-align:right"><button id="btn-test-emb" type="button" class="test-btn test-btn-idle" onclick="testEmbedding()">🔗 测试连接</button></div>

<h3>粗筛 (Coarse)</h3>
<div class="row">
  <div><label>跳过阈值 (≤N 篇时跳过粗筛)</label><input id="cfg-COARSE_POOL_SKIP_THRESHOLD" type="number" step="1"></div>
  <div class="emb-dependent"><label>Embedding 最低相似度</label><input id="cfg-COARSE_EMBEDDING_MIN_SIMILARITY" type="number" step="0.01" min="0" max="1"></div>
</div>
<div class="row">
  <div><label>BM25 权重</label><input id="cfg-COARSE_WEIGHT_BM25" type="number" step="0.01"></div>
  <div class="emb-dependent"><label>Embedding 权重</label><input id="cfg-COARSE_WEIGHT_EMBEDDING" type="number" step="0.01"></div>
  <div><label>Structure 权重</label><input id="cfg-COARSE_WEIGHT_STRUCTURE" type="number" step="0.01"></div>
</div>
<div class="row">
  <div><label>降级 BM25 权重</label><input id="cfg-COARSE_WEIGHT_BM25_FALLBACK" type="number" step="0.01"></div>
  <div><label>降级 Structure 权重</label><input id="cfg-COARSE_WEIGHT_STRUCTURE_FALLBACK" type="number" step="0.01"></div>
</div>
<div class="row">
  <div><label>截断相对阈值因子</label><input id="cfg-COARSE_RELATIVE_THRESHOLD_FACTOR" type="number" step="0.01"></div>
</div>

<h3>重排序 (Reranker)</h3>
<div class="row">
  <div><label>Top-K 保底</label><input id="cfg-RERANKER_TOP_K_FALLBACK" type="number" step="1"></div>
  <div><label>相对阈值因子</label><input id="cfg-RERANKER_RELATIVE_THRESHOLD_FACTOR" type="number" step="0.01"></div>
</div>

<h3>精筛 (Judge)</h3>
<div class="row">
  <div><label>并发数</label><input id="cfg-JUDGE_CONCURRENCY" type="number" step="1"></div>
  <div><label>单波上限</label><input id="cfg-JUDGE_WAVE_CAP" type="number" step="1"></div>
</div>

<p style="font-size:11px;color:#888;margin-top:8px">配置保存到 .env.local（本地私密，不提交 Git）。密码字段不显示当前值。</p>

<div class="actions">
  <button type="button" class="btn-secondary" onclick="closeConfig(false)">取消</button>
  <button type="submit">保存</button>
</div>
</form>
</div>
</div>

<div id="config-toast" class="toast" style="display:none">✅ 配置已保存到 .env.local</div>

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


# ============================ Test Endpoints ============================ #


@app.post("/api/test/llm", tags=["测试"], summary="测试 LLM 连通性")
async def test_llm(body: dict[str, Any] = Body({})) -> dict[str, Any]:
    """Test LLM connectivity using provided (or env) config."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        provider = body.get("provider") or os.getenv("LLM_PROVIDER", "openai")
        api_key = body.get("api_key") or os.getenv("LLM_API_KEY", "")
        model = body.get("model") or os.getenv("LLM_MODEL") or os.getenv("LLM_FAST_MODEL") or "gpt-4o-mini"
        base_url = body.get("base_url") or os.getenv("OPENAI_BASE_URL") or (
            "https://dashscope.aliyuncs.com/compatible-mode/v1" if provider == "dashscope"
            else "https://api.openai.com/v1"
        )

        llm = ChatOpenAI(model=model, temperature=0.0, api_key=api_key, base_url=base_url)
        resp = llm.invoke([HumanMessage(content="Respond with only: OK")])
        msg = resp.content.strip()[:100] if hasattr(resp, "content") else str(resp)[:100]
        return {"ok": True, "message": f"LLM 响应正常: {msg}"}
    except Exception as e:
        return {"ok": False, "message": f"LLM 测试失败: {str(e)[:200]}"}


@app.post("/api/test/embedding", tags=["测试"], summary="测试 Embedding API")
async def test_embedding(body: dict[str, Any] = Body({})) -> dict[str, Any]:
    """Test embedding API using provided (or env) config."""
    try:
        import dashscope
        from http import HTTPStatus

        api_key = body.get("api_key") or os.getenv("EMBEDDING_API_KEY", "")
        model = body.get("model") or os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        if not api_key:
            return {"ok": False, "message": "未配置 Embedding API Key"}

        resp = dashscope.TextEmbedding.call(model=model, input="test query", api_key=api_key)
        if resp.status_code == HTTPStatus.OK:
            dim = len(resp.output["embeddings"][0]["embedding"])
            return {"ok": True, "message": f"Embedding 正常，向量维度: {dim}"}
        return {"ok": False, "message": f"Embedding 返回错误: {resp.status_code}"}
    except Exception as e:
        return {"ok": False, "message": f"Embedding 测试失败: {str(e)[:200]}"}


@app.post("/api/test/reranker", tags=["测试"], summary="测试 Reranker API")
async def test_reranker(body: dict[str, Any] = Body({})) -> dict[str, Any]:
    """Test reranker API using provided (or env) config."""
    try:
        import dashscope
        from http import HTTPStatus

        api_key = body.get("api_key") or os.getenv("RERANKER_API_KEY", "")
        model = body.get("model") or os.getenv("RERANKER_MODEL", "qwen3-rerank")
        if not api_key:
            return {"ok": False, "message": "未配置 Reranker API Key"}

        resp = dashscope.TextReRank.call(
            model=model, query="test", documents=["test document"],
            top_n=1, return_documents=False, api_key=api_key,
        )
        if resp.status_code == HTTPStatus.OK:
            return {"ok": True, "message": "Reranker API 正常"}
        return {"ok": False, "message": f"Reranker 返回错误: {resp.status_code} - {resp.message}"}
    except Exception as e:
        return {"ok": False, "message": f"Reranker 测试失败: {str(e)[:200]}"}


@app.post("/api/test/semantic-scholar", tags=["测试"], summary="测试 Semantic Scholar API")
async def test_semantic_scholar(body: dict[str, Any] = Body({})) -> dict[str, Any]:
    """Test S2 API using provided (or env) key."""
    try:
        import httpx

        api_key = body.get("api_key") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        url = "https://api.semanticscholar.org/graph/v1/paper/search?query=transformer&limit=1&fields=paperId"

        with httpx.Client(trust_env=False) as client:
            headers: dict[str, str] = {"Accept": "application/json"}
            if api_key:
                headers["x-api-key"] = api_key
            print(f"[S2_DEBUG] headers={headers}", flush=True)
            resp = client.get(url, headers=headers, timeout=15.0)
            print(f"[S2_DEBUG] status={resp.status_code} headers={dict(resp.headers)} body={resp.text[:200]}", flush=True)
            resp.raise_for_status()
            data = resp.json()
            total = data.get("total", 0)
        return {"ok": True, "message": f"S2 API 正常，搜索到 {total} 篇论文"}
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:150] if e.response.text else ""
        debug_key = (api_key[:8] + "...") if api_key else "(empty)"
        return {"ok": False, "message": f"S2 请求失败 (HTTP {e.response.status_code}): {detail} [key={debug_key}]"}
    except Exception as e:
        return {"ok": False, "message": f"S2 测试失败: {str(e)[:200]}"}


@app.post("/api/test/openalex", tags=["测试"], summary="测试 OpenAlex API")
async def test_openalex(body: dict[str, Any] = Body({})) -> dict[str, Any]:
    """Test OA API using provided (or env) mailto."""
    try:
        import httpx

        mailto = body.get("mailto") or os.getenv("OPENALEX_MAILTO")
        url = "https://api.openalex.org/works?search=machine+learning&per_page=1"
        if mailto:
            url += f"&mailto={mailto}"
        resp = httpx.get(url, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        count = data.get("meta", {}).get("count", 0)
        return {"ok": True, "message": f"OA API 正常，搜索到 {count:,} 篇论文"}
    except httpx.HTTPStatusError as e:
        return {"ok": False, "message": f"OA 请求失败 (HTTP {e.response.status_code})"}
    except Exception as e:
        return {"ok": False, "message": f"OA 测试失败: {str(e)[:200]}"}


def main() -> None:
    import uvicorn
    uvicorn.run("paper_search.web:app", host="127.0.0.1", port=8000, reload=False)
