import json
import os

paths = [
    (
        "/root/autodl-tmp/application/database/dbguarddog/outputs/eval_results/qwen_binary_lr8e6_steps330_expanded50_results.json",
        "database",
    ),
    (
        "/root/autodl-tmp/application/email/mailguarddog/outputs/eval_results/qwen_binary_lr8e6_steps330_expanded50_results.json",
        "email",
    ),
]

for path, name in paths:
    print(f"=== {name} ===")
    print("exists", os.path.exists(path), "size", os.path.getsize(path) if os.path.exists(path) else 0)
    if not os.path.exists(path):
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(json.dumps(data.get("metrics", {}), ensure_ascii=False, indent=2))
    cases = data.get("cases", [])
    print("cases", len(cases))

    mismatches = []
    for row in cases:
        bad = []
        if row.get("decision") != row.get("expected_decision"):
            bad.append("decision")
        if row.get("verdict") != row.get("expected_verdict"):
            bad.append("verdict")
        if bad:
            mismatches.append(
                (
                    row.get("case_id"),
                    bad,
                    row.get("decision"),
                    row.get("verdict"),
                    row.get("expected_decision"),
                    row.get("expected_verdict"),
                )
            )

    print("mismatches", len(mismatches))
    for item in mismatches[:20]:
        print(item)
