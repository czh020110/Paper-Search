from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_settings
from .pipeline import run_pipeline
from .web import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper search minimal offline loop")
    parser.add_argument("--query", help="Natural language academic query")
    parser.add_argument("--backend", default="mock", choices=["mock"], help="Retrieval backend")
    parser.add_argument("--output-dir", default=None, help="Override output root directory")
    parser.add_argument("--serve", action="store_true", help="Start the local web interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local web interface")
    parser.add_argument("--port", type=int, default=8000, help="Port for the local web interface")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings()
    output_dir = Path(args.output_dir) if args.output_dir else settings.output_dir

    if args.serve:
        run_server(host=args.host, port=args.port, output_root=output_dir, settings=settings)
        return 0

    if not args.query:
        parser.error("--query is required unless --serve is used")

    result = run_pipeline(query=args.query, backend=args.backend, output_root=output_dir, settings=settings)
    print(json.dumps(_cli_summary(result), ensure_ascii=False, indent=2))
    return 0


def _cli_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": result["run_id"],
        "paper_count": len(result["papers"]),
        "query_type": result["query_plan"]["intent_analysis"]["query_type"],
        "output_files": result["output_files"],
        "evaluation": result["evaluation"],
    }
