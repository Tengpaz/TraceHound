import json
import shutil
from pathlib import Path

from traceguard.data import built_in_cases
from traceguard.demo_app import _list_eval_datasets, _load_batch_cases_from_dataset


def test_demo_can_select_existing_generated_eval_dataset():
    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "data/tmp/pytest_demo_eval_dataset"
    dataset_path = dataset_dir / "synthetic_eval.jsonl"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        with dataset_path.open("w", encoding="utf-8") as handle:
            for case in built_in_cases(count=4):
                handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

        datasets = _list_eval_datasets(root)
        paths = {item["path"] for item in datasets}
        assert "data/tmp/pytest_demo_eval_dataset/synthetic_eval.jsonl" in paths

        rows = _load_batch_cases_from_dataset(
            root,
            "data/tmp/pytest_demo_eval_dataset/synthetic_eval.jsonl",
        )
        assert len(rows) == 4
        assert all(gold is not None for _, gold in rows)
    finally:
        shutil.rmtree(dataset_dir, ignore_errors=True)
