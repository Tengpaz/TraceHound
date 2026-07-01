import json

from traceguard.data import built_in_cases
from traceguard.dataset_layout import write_dataset_bundle
from traceguard.production import filter_cases_for_training


def test_write_dataset_bundle_creates_clean_layout_and_legacy_files(tmp_path):
    cases = built_in_cases(count=24)
    training_cases, _ = filter_cases_for_training(cases)

    bundle = write_dataset_bundle(tmp_path, cases, training_cases, include_rl=True)

    expected_files = [
        "cases/all.jsonl",
        "cases/train.jsonl",
        "cases/eval.jsonl",
        "cases/test.jsonl",
        "train/agentdog/binary_safety/train.jsonl",
        "train/agentdog/taxonomy_only/all.jsonl",
        "train/agentdog/unified_four_label/all.jsonl",
        "train/rl/rl_pairs/all.jsonl",
        "metadata/coverage_matrix.json",
        "metadata/dataset_manifest.json",
        "synthetic_eval.jsonl",
        "agentdog_unified_sft.jsonl",
        "synthetic_rl.jsonl",
    ]
    for relative in expected_files:
        assert (tmp_path / relative).exists(), relative

    manifest = json.loads((tmp_path / "metadata" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "tracehound.dataset_bundle.v1"
    assert manifest["splits"]["counts"] == bundle["splits"]["counts"]
    assert bundle["coverage"]["all_cases"]["samples"] == 24
    assert bundle["counts"]["cases/all.jsonl"] == 24
