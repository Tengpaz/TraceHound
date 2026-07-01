from traceguard.data import TOOL_SCENARIOS, built_in_cases
from traceguard.dataset_ops import coverage_matrix, split_cases, split_summary
from traceguard.taxonomy import FAILURE_MODES, HARM_TYPES, RISK_SOURCES


def test_coverage_matrix_reports_agentdog_axes():
    cases = built_in_cases(count=224, labels=["unsafe"])

    coverage = coverage_matrix(cases)

    assert coverage["samples"] == 224
    assert coverage["unsafe_samples"] == 224
    assert coverage["coverage"]["tool_scenarios"]["present"] == len(TOOL_SCENARIOS)
    assert coverage["coverage"]["risk_sources"]["present"] == len(RISK_SOURCES)
    assert coverage["coverage"]["failure_modes"]["present"] == len(FAILURE_MODES)
    assert coverage["coverage"]["harm_types"]["present"] == len(HARM_TYPES)
    assert coverage["missing"]["risk_sources"] == []
    assert set(coverage["matrices"]["risk_source_by_failure_mode"]) == set(RISK_SOURCES)
    assert coverage["taxonomy_triples"]


def test_split_cases_is_deterministic_and_complete():
    cases = built_in_cases(count=80)

    first = split_cases(cases, train_ratio=0.75, eval_ratio=0.15, test_ratio=0.10, seed=123)
    second = split_cases(cases, train_ratio=0.75, eval_ratio=0.15, test_ratio=0.10, seed=123)

    assert {name: [case["id"] for case in rows] for name, rows in first.items()} == {
        name: [case["id"] for case in rows] for name, rows in second.items()
    }
    split_ids = [case["id"] for rows in first.values() for case in rows]
    assert sorted(split_ids) == sorted(case["id"] for case in cases)
    assert len(split_ids) == len(set(split_ids))
    assert all(first[name] for name in ("train", "eval", "test"))
    summary = split_summary(first, train_ratio=0.75, eval_ratio=0.15, test_ratio=0.10, seed=123)
    assert summary["counts"]["train"] + summary["counts"]["eval"] + summary["counts"]["test"] == 80
    assert summary["seed"] == 123
