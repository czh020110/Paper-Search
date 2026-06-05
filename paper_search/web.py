from __future__ import annotations

import html
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from .config import Settings, load_settings
from .pipeline import run_pipeline


def run_server(host: str, port: int, output_root: Path, settings: Settings) -> None:
    handler = _build_handler(output_root=output_root, settings=settings)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Paper Search Web is running at http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _build_handler(output_root: Path, settings: Settings):
    class PaperSearchHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._send_html(_render_page())

        def do_POST(self) -> None:
            if self.path != "/search":
                self.send_error(404)
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            form_data = parse_qs(raw_body)
            query = (form_data.get("query") or [""])[0].strip()
            if not query:
                self._send_html(_render_page(error="请输入查询内容。"), status=400)
                return

            try:
                result = run_pipeline(query=query, backend="live", output_root=output_root, settings=settings)
                self._send_html(_render_page(query=query, result=result))
            except Exception as exc:  # noqa: BLE001
                self._send_html(_render_page(query=query, error=str(exc)), status=500)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return None

        def _send_html(self, body: str, status: int = 200) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return PaperSearchHandler


def _render_page(query: str = "", result: dict[str, object] | None = None, error: str | None = None) -> str:
    escaped_query = html.escape(query)
    result_block = ""
    if result:
        result_block = _render_result_block(result)
    error_block = f'<p class="error">{html.escape(error)}</p>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Paper Search</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 32px auto; padding: 0 16px; }}
    form {{ display: flex; gap: 12px; margin-bottom: 24px; }}
    input[type=text] {{ flex: 1; padding: 10px; font-size: 14px; }}
    button {{ padding: 10px 16px; cursor: pointer; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    .error {{ color: #b91c1c; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    code {{ background: #f5f5f5; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Paper Search 最小闭环</h1>
  <p>当前页面调用 Semantic Scholar 与 OpenAlex 真实学术检索 API，查询理解 → 初检索 → 去重 → 结果整理。</p>
  <form method="post" action="/search">
    <input type="text" name="query" value="{escaped_query}" placeholder="输入学术查询，例如：2022年后关于大模型幻觉控制的CVPR强化学习论文" />
    <button type="submit">运行检索</button>
  </form>
  {error_block}
  {result_block}
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the Paper Search web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    settings = load_settings()
    output_root = Path(args.output_dir) if args.output_dir else settings.output_dir
    run_server(host=args.host, port=args.port, output_root=output_root, settings=settings)
    return 0


def _render_result_block(result: dict[str, object]) -> str:
    query_plan = result["query_plan"]
    papers = result["papers"]
    output_files = result["output_files"]
    evaluation = result["evaluation"]

    paper_rows = "".join(
        f"<tr><td>{html.escape(str(paper['title']))}</td><td>{html.escape(str(paper.get('year', '')))}</td><td>{html.escape(str(paper.get('venue', '')))}</td><td>{html.escape(str(paper.get('source_api', '')))}</td></tr>"
        for paper in papers
    )

    return f"""
  <div class="card">
    <h2>运行结果</h2>
    <p><strong>Run ID：</strong><code>{html.escape(str(result['run_id']))}</code></p>
    <p><strong>Query Type：</strong>{html.escape(str(query_plan['intent_analysis']['query_type']))}</p>
    <p><strong>命中文献数：</strong>{len(papers)}</p>
    <p><strong>Precision / Recall / F1：</strong>{html.escape(str(evaluation.get('precision')))} / {html.escape(str(evaluation.get('recall')))} / {html.escape(str(evaluation.get('f1')))}</p>
    <p><strong>输出文件：</strong></p>
    <ul>
      <li>result.md：<code>{html.escape(str(output_files['markdown']))}</code></li>
      <li>graph.json：<code>{html.escape(str(output_files['graph']))}</code></li>
      <li>experiment.json：<code>{html.escape(str(output_files['experiment']))}</code></li>
    </ul>
    <table>
      <thead><tr><th>标题</th><th>年份</th><th>Venue</th><th>来源</th></tr></thead>
      <tbody>{paper_rows}</tbody>
    </table>
  </div>
"""
