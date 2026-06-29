#!/usr/bin/env python
"""List configured local/API model profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.model_profiles import load_model_profiles, profile_summary, resolve_model_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Show one profile in full.")
    parser.add_argument("--path", help="Optional model profile JSON path.")
    parser.add_argument("--full", action="store_true", help="Print full profile objects.")
    args = parser.parse_args()

    if args.profile:
        payload = resolve_model_profile(args.profile, args.path)
    else:
        data = load_model_profiles(args.path)
        payload = {
            "default_local_profile": data.get("default_local_profile"),
            "default_api_profile": data.get("default_api_profile"),
            "profiles": data.get("profiles", {}) if args.full else [
                profile_summary({"name": name, **profile})
                for name, profile in sorted(data.get("profiles", {}).items())
            ],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
