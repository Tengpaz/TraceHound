#!/usr/bin/env python
"""Serve the TraceHound FastAPI demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.config import load_env_file
from traceguard.demo_app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    load_env_file(args.env_file)

    try:
        import uvicorn
    except ModuleNotFoundError:
        from traceguard.fallback_server import run_fallback_server

        print(
            "uvicorn is not installed; using the dependency-free fallback server. "
            "For FastAPI mode, run `conda env create -f environment.yml` and `conda activate tracehound`."
        )
        run_fallback_server(args.host, args.port)
        return

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
