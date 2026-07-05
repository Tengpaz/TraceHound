#!/usr/bin/env python
"""Sample safe AgentDoG BinarySafety rows and expand them with JSON API labels."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data/datasets/agentdog_raw/AgentDoG-BinarySafety/train.json"
DEFAULT_PROMPT_PATH = REPO_ROOT / "prompts/security_analyst_json_prompt.txt"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs/data/agentdog_binary_safe_dedup1000_json_augmented.jsonl"
DEFAULT_FAILED_PATH = REPO_ROOT / "outputs/data/agentdog_binary_safe_dedup1000_json_augmented_failed.jsonl"
DEFAULT_SAMPLED_PATH = REPO_ROOT / "outputs/data/agentdog_binary_safe_dedup1000_sampled.jsonl"

DATASET_ID = "AI45Research/AgentDoG1.0-Training-Data"
DATASET_SUBSET = "AgentDoG-BinarySafety"
REQUIRED_KEYS = ("risk_source", "failure_mode", "harm_type", "rationale", "source", "judgment")
SOURCE_VALUES = {"benign", "safe", "unsafe", "false_refusal"}
JUDGMENT_VALUES = {"safe", "unsafe"}
RISK_SOURCE_VALUES = {
    "benign",
    "malicious_user_instruction_or_jailbreak",
    "direct_prompt_injection",
    "indirect_prompt_injection",
    "unreliable_or_misinformation",
    "tool_description_injection",
    "malicious_tool_execution",
    "corrupted_tool_feedback",
    "inherent_agent_failures",
}
FAILURE_MODE_VALUES = {
    "benign",
    "unconfirmed_or_overprivileged_action",
    "flawed_planning_or_reasoning",
    "incorrect_tool_parameters",
    "choosing_malicious_tool",
    "tool_misuse_in_specific_context",
    "failure_to_validate_tool_outputs",
    "insecure_interaction_or_execution",
    "procedural_deviation_or_inaction",
    "inefficient_or_wasteful_execution",
    "generation_of_harmful_offensive_content",
    "instruction_for_harmful_illegal_activity",
    "generation_of_malicious_executables",
    "unauthorized_information_disclosure",
    "provide_inaccurate_misleading_or_unverified_information",
}
HARM_TYPE_VALUES = {
    "benign",
    "privacy_and_confidentiality_harm",
    "financial_and_economic_harm",
    "security_and_system_integrity_harm",
    "physical_and_health_harm",
    "psychological_and_emotional_harm",
    "reputation_and_interpersonal_harm",
    "info_ecosystem_and_societal_harm",
    "public_service_and_resource_harm",
    "fairness_equity_and_allocative_harm",
    "functional_and_opportunity_harm",
}


class ApiError(RuntimeError):
    """Raised when the OpenAI-compatible API call fails."""


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args)
    examples = load_json_list(dataset_path)
    safe_examples = [(idx, row) for idx, row in enumerate(examples) if str(row.get("output", "")).strip() == "safe"]
    original_safe_count = len(safe_examples)
    if args.dedupe:
        safe_examples = dedupe_examples(safe_examples)
        print(
            f"[dedupe] safe_rows={original_safe_count} unique_safe_rows={len(safe_examples)} "
            f"removed={original_safe_count - len(safe_examples)}",
            flush=True,
        )
    if len(safe_examples) < args.max_samples:
        raise SystemExit(f"Need {args.max_samples} safe examples, but found only {len(safe_examples)} in {dataset_path}")

    sampled = sample_examples(safe_examples, max_samples=args.max_samples, seed=args.seed)
    sampled = sampled[args.start :]
    if args.limit:
        sampled = sampled[: args.limit]

    prompt_path = Path(args.prompt_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    if "{formatted_trajectory}" not in prompt_template:
        raise SystemExit("Prompt template must contain the literal placeholder {formatted_trajectory}")

    write_sampled_rows(args.sampled_path, sampled, prompt_template, args.write_sampled)
    if args.prepare_only:
        print_summary("prepared", dataset_path, args.output_path, sampled, written=0, failed=0)
        return

    client = ChatClient(
        model=args.model,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        api_path=args.api_path,
        timeout=args.timeout,
    )
    completed = read_completed_source_indices(args.output_path)
    output_path = Path(args.output_path)
    failed_path = Path(args.failed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    pending = [
        (ordinal, source_index, example)
        for ordinal, (source_index, example) in enumerate(sampled, start=1)
        if source_index not in completed
    ]
    already_done = len(sampled) - len(pending)
    if already_done:
        print(f"[resume] skipped_completed={already_done} pending={len(pending)} total={len(sampled)}", flush=True)

    written = 0
    failed = 0
    with output_path.open("a", encoding="utf-8") as out_f, failed_path.open("a", encoding="utf-8") as fail_f:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            future_to_item = {
                executor.submit(
                    build_augmented_row,
                    source_index=source_index,
                    example=example,
                    prompt_template=prompt_template,
                    client=client,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    sleep_seconds=args.sleep_seconds,
                    prompt_path=prompt_path,
                    rationale_min_chars=args.rationale_min_chars,
                    rationale_max_chars=args.rationale_max_chars,
                    include_analysis_json=args.include_analysis_json,
                    include_raw_response=args.include_raw_response,
                ): (ordinal, source_index, example)
                for ordinal, source_index, example in pending
            }
            processed = already_done
            for future in as_completed(future_to_item):
                ordinal, source_index, example = future_to_item[future]
                processed += 1
                try:
                    row = future.result()
                except Exception as exc:
                    failed += 1
                    append_jsonl_row(fail_f, build_failed_row(source_index, example, exc), fsync=not args.no_fsync_each_row)
                    print_progress("failed", processed, len(sampled), source_index, str(exc)[:180])
                    continue
                written += 1
                completed.add(source_index)
                append_jsonl_row(out_f, row, fsync=not args.no_fsync_each_row)
                print_progress("ok", processed, len(sampled), source_index, row["metadata"]["source"])

    print_summary("done", dataset_path, output_path, sampled, written=written, failed=failed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--download-if-missing", action="store_true", help="Download the Hugging Face dataset snapshot if --dataset-path is absent.")
    parser.add_argument("--download-root", type=Path, default=REPO_ROOT / "data/datasets/agentdog_raw/AgentDoG1.0-Training-Data")
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--failed-path", type=Path, default=DEFAULT_FAILED_PATH)
    parser.add_argument("--sampled-path", type=Path, default=DEFAULT_SAMPLED_PATH)
    parser.add_argument("--write-sampled", action="store_true", help="Write the selected 1000 safe source rows before API generation.")
    parser.add_argument("--prepare-only", action="store_true", help="Only sample and optionally write source rows; do not call the API.")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent API requests.")
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True, help="Deduplicate source rows before sampling.")
    parser.add_argument("--start", type=int, default=0, help="Skip this many sampled rows before processing.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many rows after --start; 0 means no extra limit.")
    parser.add_argument("--model", default=os.environ.get("MODEL", "gpt-4.1-mini"))
    parser.add_argument("--api-base", default=os.environ.get("API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--api-path", default=os.environ.get("API_PATH", "/chat/completions"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--rationale-min-chars", type=int, default=80)
    parser.add_argument("--rationale-max-chars", type=int, default=700)
    parser.add_argument("--include-analysis-json", action="store_true", help="Also store the parsed JSON outside messages for analysis.")
    parser.add_argument("--include-raw-response", action="store_true", help="Also store raw model text. Off by default to avoid redundant output.")
    parser.add_argument("--no-fsync-each-row", action="store_true", help="Disable fsync after each JSONL row.")
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path.exists():
        return args.dataset_path
    if not args.download_if_missing:
        raise SystemExit(f"Dataset not found: {args.dataset_path}. Use --download-if-missing or pass --dataset-path.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub or use the existing local train.json path.") from exc

    args.download_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(args.download_root),
        allow_patterns=[
            f"{DATASET_SUBSET}/train.json",
            f"{DATASET_SUBSET}/*.json",
            "train.json",
            "*.json",
        ],
    )
    for candidate in (
        args.download_root / DATASET_SUBSET / "train.json",
        args.download_root / "train.json",
    ):
        if candidate.exists():
            return candidate
    matches = sorted(args.download_root.glob(f"**/{DATASET_SUBSET}*/train.json"))
    if matches:
        return matches[0]
    raise SystemExit(f"Downloaded {DATASET_ID}, but could not find {DATASET_SUBSET}/train.json under {args.download_root}")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def sample_examples(rows: list[tuple[int, dict[str, Any]]], *, max_samples: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)
    return selected[:max_samples]


def dedupe_examples(rows: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
    seen: set[str] = set()
    unique: list[tuple[int, dict[str, Any]]] = []
    for source_index, row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append((source_index, row))
    return unique


def write_sampled_rows(
    path: Path,
    sampled: list[tuple[int, dict[str, Any]]],
    prompt_template: str,
    enabled: bool,
) -> None:
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ordinal, (source_index, example) in enumerate(sampled, start=1):
            trajectory = extract_trajectory(example)
            prompt = render_prompt(prompt_template, trajectory)
            row = {
                "id": f"agentdog-binary-safe-{source_index:06d}",
                "sample_ordinal": ordinal,
                "source_index": source_index,
                "source_label": "safe",
                "trajectory": trajectory,
                "messages": [{"role": "user", "content": prompt}],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class ChatClient:
    def __init__(self, *, model: str, api_base: str, api_key_env: str, api_path: str, timeout: int) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_path = api_path if api_path.startswith("/") else "/" + api_path
        self.timeout = timeout
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ApiError(f"Missing API key environment variable: {api_key_env}")

    def complete_json(self, prompt: str, *, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ApiError(f"HTTP {exc.code}: {body[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise ApiError(f"request failed: {exc}") from exc
        return extract_message_content(raw)

    @property
    def endpoint(self) -> str:
        if self.api_base.endswith("/chat/completions"):
            return self.api_base
        return self.api_base + self.api_path


def build_augmented_row(
    *,
    source_index: int,
    example: dict[str, Any],
    prompt_template: str,
    client: ChatClient,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
    prompt_path: Path,
    rationale_min_chars: int,
    rationale_max_chars: int,
    include_analysis_json: bool,
    include_raw_response: bool,
) -> dict[str, Any]:
    trajectory = extract_trajectory(example)
    prompt = render_prompt(prompt_template, trajectory)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            raw_response = client.complete_json(prompt, temperature=temperature)
            analysis_json = validate_analysis_json(
                parse_json_object(raw_response),
                rationale_min_chars=rationale_min_chars,
                rationale_max_chars=rationale_max_chars,
            )
            validate_safe_source_assumption(analysis_json)
            assistant_content = json.dumps(analysis_json, ensure_ascii=False, sort_keys=True)
            row = {
                "id": f"agentdog-binary-safe-{source_index:06d}",
                "source_dataset": DATASET_ID,
                "source_subset": DATASET_SUBSET,
                "source_index": source_index,
                "source_label": "safe",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "generation_method": "api_json_expansion",
                    "model": client.model,
                    "api_base": redact_api_base(client.api_base),
                    "prompt_path": str(prompt_path),
                    "attempt": attempt,
                    "source": analysis_json["source"],
                    "judgment": analysis_json["judgment"],
                },
            }
            if include_analysis_json:
                row["analysis_json"] = analysis_json
            if include_raw_response:
                row["raw_response"] = raw_response
            return row
        except Exception as exc:
            last_error = exc
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def extract_trajectory(example: dict[str, Any]) -> str:
    text = str(example.get("instruction", ""))
    if example.get("input"):
        text = f"{text}\n\n{example['input']}"
    match = re.search(r"<BEGIN TRAJECTORY>\s*(.*?)\s*<END TRAJECTORY>", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find <BEGIN TRAJECTORY> block")
    return match.group(1).strip()


def render_prompt(template: str, trajectory: str) -> str:
    return template.replace("{formatted_trajectory}", trajectory)


def extract_message_content(raw: dict[str, Any]) -> str:
    if isinstance(raw.get("output_text"), str):
        return raw["output_text"]
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                if any(parts):
                    return "\n".join(parts)
        if isinstance(choices[0], dict) and isinstance(choices[0].get("text"), str):
            return choices[0]["text"]
    raise ApiError("API response does not contain a supported text field")


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("API output is not a JSON object")
    return parsed


def validate_analysis_json(
    value: dict[str, Any],
    *,
    rationale_min_chars: int,
    rationale_max_chars: int,
) -> dict[str, str]:
    missing = [key for key in REQUIRED_KEYS if key not in value]
    if missing:
        raise ValueError(f"Missing JSON keys: {missing}")
    extra = [key for key in value if key not in REQUIRED_KEYS]
    if extra:
        raise ValueError(f"Unexpected JSON keys: {extra}")
    normalized = {key: normalize_label_value(key, str(value[key]).strip()) for key in REQUIRED_KEYS}
    if normalized["source"] not in SOURCE_VALUES:
        raise ValueError(f"Invalid source: {normalized['source']}")
    if normalized["judgment"] not in JUDGMENT_VALUES:
        raise ValueError(f"Invalid judgment: {normalized['judgment']}")
    if normalized["risk_source"] not in RISK_SOURCE_VALUES:
        raise ValueError(f"Invalid risk_source: {normalized['risk_source']}")
    if normalized["failure_mode"] not in FAILURE_MODE_VALUES:
        raise ValueError(f"Invalid failure_mode: {normalized['failure_mode']}")
    if normalized["harm_type"] not in HARM_TYPE_VALUES:
        raise ValueError(f"Invalid harm_type: {normalized['harm_type']}")
    validate_rationale(
        normalized["rationale"],
        min_chars=rationale_min_chars,
        max_chars=rationale_max_chars,
    )
    if normalized["source"] == "benign":
        for key in ("risk_source", "failure_mode", "harm_type"):
            if normalized[key] != "benign":
                raise ValueError(f"source=benign requires {key}=benign")
        if normalized["judgment"] != "safe":
            raise ValueError("source=benign requires judgment=safe")
    if normalized["source"] == "safe" and normalized["judgment"] != "safe":
        raise ValueError("source=safe requires judgment=safe")
    if normalized["source"] in {"unsafe", "false_refusal"}:
        if normalized["judgment"] != "unsafe":
            raise ValueError("unsafe/false_refusal source requires judgment=unsafe")
        for key in ("risk_source", "failure_mode", "harm_type"):
            if normalized[key] == "benign":
                raise ValueError(f"{normalized['source']} requires {key} not to be benign")
    return normalized


def normalize_label_value(key: str, value: str) -> str:
    if key in {"risk_source", "failure_mode", "harm_type"} and value == "Benign":
        return "benign"
    return value


def validate_rationale(rationale: str, *, min_chars: int, max_chars: int) -> None:
    compact = re.sub(r"\s+", " ", rationale).strip()
    if len(compact) < min_chars:
        raise ValueError(f"rationale is too short: {len(compact)} chars")
    if len(compact) > max_chars:
        raise ValueError(f"rationale is too long: {len(compact)} chars")
    lowered = compact.lower()
    forbidden = ("<think", "</think", "chain-of-thought", "```")
    if any(token in lowered for token in forbidden):
        raise ValueError("rationale contains hidden-reasoning or markdown markers")
    repeated_prefixes = (
        "risk_source:",
        "failure_mode:",
        "harm_type:",
        "source:",
        "judgment:",
    )
    if sum(lowered.count(prefix) for prefix in repeated_prefixes) >= 2:
        raise ValueError("rationale redundantly repeats output field labels")


def validate_safe_source_assumption(value: dict[str, str]) -> None:
    if value["judgment"] != "safe" or value["source"] not in {"benign", "safe"}:
        raise ValueError(
            "The source AgentDoG row is labeled safe, but the generated JSON classified it as "
            f"source={value['source']} judgment={value['judgment']}"
        )


def read_completed_source_indices(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed.add(int(row["source_index"]))
            except Exception:
                continue
    return completed


def build_failed_row(source_index: int, example: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "id": f"agentdog-binary-safe-{source_index:06d}",
        "source_index": source_index,
        "source_label": str(example.get("output", "")),
        "error": str(exc),
    }


def append_jsonl_row(handle: Any, row: dict[str, Any], *, fsync: bool) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    if fsync:
        os.fsync(handle.fileno())


def print_progress(status: str, current: int, total: int, source_index: int, detail: str) -> None:
    print(f"[{status}] {current}/{total} source_index={source_index} {detail}", flush=True)


def print_summary(status: str, dataset_path: Path, output_path: Path, sampled: list[tuple[int, dict[str, Any]]], *, written: int, failed: int) -> None:
    summary = {
        "status": status,
        "dataset_path": str(dataset_path),
        "sampled": len(sampled),
        "written": written,
        "failed": failed,
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def redact_api_base(api_base: str) -> str:
    return api_base.rstrip("/").split("://", 1)[-1].split("/", 1)[0]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
