"""Optional local Hugging Face model adapter.

This module is intentionally dependency-light at import time. Transformers,
torch, and PEFT-related packages are imported only when the local adapter is
instantiated on a GPU-capable environment.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from traceguard.chat_format import apply_chat_template
from traceguard.json_utils import extract_json_object
from traceguard.judge import ModelAdapter
from traceguard.lite_binary_eval import build_lite_binary_prompt, parse_judgment_output
from traceguard.model_profiles import profile_model_id, resolve_model_profile
from traceguard.prompts import build_remote_messages
from traceguard.schema import CostStats, RiskReport, TrajectoryCase, TrajectoryStep


DEFAULT_BINARY_GUARD_PROFILE = "tracehound-base-qwen3_5-0_8b-binary"
ROOT = Path(__file__).resolve().parents[1]


class LocalTransformersAdapter(ModelAdapter):
    """Local CausalLM adapter for InternLM and other chat-template models."""

    def __init__(
        self,
        *,
        model_profile: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        profile_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        trust_remote_code: Optional[bool] = None,
        device_map: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local model inference requires optional GPU dependencies. "
                "On the Linux/GPU server, install CUDA-matched PyTorch first, then run `pip install -e \".[train]\"`."
            ) from exc

        self.profile = resolve_model_profile(model_profile, profile_path)
        if self.profile.get("provider") != "huggingface":
            raise ValueError(f"profile {self.profile['name']} is not a Hugging Face local model profile")
        self.model_name_or_path = _resolve_model_reference(
            model_name_or_path
            or os.getenv("TRACEHOUND_LOCAL_MODEL_PATH")
            or profile_model_id(self.profile)
        )
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path is required for local inference")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trust_remote_code = (
            bool(self.profile.get("trust_remote_code", False)) if trust_remote_code is None else trust_remote_code
        )
        self.device_map = device_map or str(self.profile.get("device_map") or "auto")
        dtype_name = torch_dtype or str(self.profile.get("torch_dtype") or "auto")
        dtype = _resolve_torch_dtype(torch, dtype_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            torch_dtype=dtype,
        )
        self.model.eval()

    def evaluate(self, case: TrajectoryCase, mode: str = "layered") -> RiskReport:
        import torch  # type: ignore

        prompt_mode = "full" if mode == "full" else "compressed"
        messages = build_remote_messages(case, mode=prompt_mode)
        prompt = apply_chat_template(self.tokenizer, messages, add_generation_prompt=True)
        started = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        device = _first_model_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)
        new_tokens = output_ids[0][input_tokens:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = extract_json_object(content)
        report = RiskReport.model_validate(parsed)
        report.cost = CostStats(
            input_tokens=input_tokens,
            output_tokens=int(new_tokens.shape[-1]),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_calls=1,
            strategy=f"local:{self.profile['name']}:{prompt_mode}",
        )
        return report


class LocalBinaryClassificationAdapter(ModelAdapter):
    """Local CausalLM adapter for TraceHound binary safe/unsafe checkpoints."""

    def __init__(
        self,
        *,
        model_profile: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        profile_path: Optional[str] = None,
        max_new_tokens: int = 32,
        max_input_tokens: Optional[int] = None,
        temperature: float = 0.0,
        trust_remote_code: Optional[bool] = None,
        device_map: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local binary guard inference requires optional GPU dependencies. "
                "On the Linux/GPU server, install CUDA-matched PyTorch first, then run `pip install -e \".[train]\"`."
            ) from exc

        profile_name = model_profile or os.getenv("TRACEHOUND_GUARD_MODEL_PROFILE") or DEFAULT_BINARY_GUARD_PROFILE
        self.profile = resolve_model_profile(profile_name, profile_path)
        if self.profile.get("provider") != "huggingface":
            raise ValueError(f"profile {self.profile['name']} is not a Hugging Face local model profile")
        self.model_name_or_path = _resolve_model_reference(
            model_name_or_path
            or os.getenv("TRACEHOUND_GUARD_MODEL_PATH")
            or os.getenv("TRACEHOUND_LOCAL_MODEL_PATH")
            or profile_model_id(self.profile)
        )
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path is required for local binary guard inference")
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens or int(self.profile.get("recommended_max_input_tokens") or 4096)
        self.temperature = temperature
        self.trust_remote_code = (
            bool(self.profile.get("trust_remote_code", False)) if trust_remote_code is None else trust_remote_code
        )
        self.device_map = device_map or str(self.profile.get("device_map") or "auto")
        dtype_name = torch_dtype or str(self.profile.get("torch_dtype") or "auto")
        dtype = _resolve_torch_dtype(torch, dtype_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            torch_dtype=dtype,
        )
        self.model.eval()

    def evaluate(self, case: TrajectoryCase, mode: str = "layered") -> RiskReport:
        import torch  # type: ignore

        trajectory_text = case_to_lite_binary_trajectory(case)
        messages = [{"role": "user", "content": build_lite_binary_prompt(trajectory_text)}]
        prompt = apply_chat_template(self.tokenizer, messages, add_generation_prompt=True)
        started = time.perf_counter()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = _first_model_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)
        new_tokens = output_ids[0][input_tokens:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        judgment, invalid = parse_judgment_output(content)
        report = risk_report_from_binary_judgment(
            judgment,
            invalid=invalid,
            raw_output=content,
            model_name=str(self.profile["name"]),
        )
        report.cost = CostStats(
            input_tokens=input_tokens,
            output_tokens=int(new_tokens.shape[-1]),
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_calls=1,
            strategy=f"local-binary:{self.profile['name']}:{mode}",
        )
        return report


def build_local_judge(
    *,
    model_profile: Optional[str] = None,
    model_path: Optional[str] = None,
    profile_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> ModelAdapter:
    profile = resolve_model_profile(model_profile, profile_path)
    if _profile_is_binary_guard(profile):
        return LocalBinaryClassificationAdapter(
            model_profile=model_profile,
            model_name_or_path=model_path,
            profile_path=profile_path,
            max_new_tokens=min(max_new_tokens, 64),
            temperature=temperature,
        )
    return LocalTransformersAdapter(
        model_profile=model_profile,
        model_name_or_path=model_path,
        profile_path=profile_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def build_local_binary_judge(
    *,
    model_profile: Optional[str] = None,
    model_path: Optional[str] = None,
    profile_path: Optional[str] = None,
    max_new_tokens: int = 32,
    max_input_tokens: Optional[int] = None,
    temperature: float = 0.0,
) -> LocalBinaryClassificationAdapter:
    return LocalBinaryClassificationAdapter(
        model_profile=model_profile,
        model_name_or_path=model_path,
        profile_path=profile_path,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        temperature=temperature,
    )


def case_to_lite_binary_trajectory(case: TrajectoryCase) -> str:
    parts: list[str] = []
    if case.task:
        parts.extend(["=== Task ===", case.task.strip(), ""])
    if case.metadata:
        summary = {
            key: value
            for key, value in case.metadata.items()
            if key in {"scenario", "platform", "event_type", "source", "cwd"} and value not in (None, "")
        }
        if summary:
            parts.extend(["=== Metadata ===", _compact_json(summary), ""])
    parts.extend(["=== Conversation History ===", ""])
    if not case.trajectory:
        parts.append("[EMPTY_TRAJECTORY]")
        return "\n".join(parts).strip()
    for step in case.trajectory:
        parts.append(_format_binary_step(step))
        parts.append("")
    return "\n".join(parts).strip()


def risk_report_from_binary_judgment(
    judgment: str | None,
    *,
    invalid: bool = False,
    raw_output: str = "",
    model_name: str = DEFAULT_BINARY_GUARD_PROFILE,
) -> RiskReport:
    if invalid or judgment not in {"safe", "unsafe"}:
        return RiskReport(
            label="unsafe",
            decision="ask_confirm",
            reason="Local binary guard returned an invalid judgment; require confirmation before continuing.",
            confidence=0.35,
            model=model_name,
            binary_judgment=judgment,
            invalid_output=True,
            raw_output=raw_output,
        )
    if judgment == "unsafe":
        return RiskReport(
            label="unsafe",
            decision="block",
            reason="Local TraceHound-Base binary guard judged the trajectory unsafe.",
            confidence=0.86,
            model=model_name,
            binary_judgment=judgment,
            raw_output=raw_output,
        )
    return RiskReport(
        label="safe",
        decision="allow",
        reason="Local TraceHound-Base binary guard judged the trajectory safe.",
        confidence=0.86,
        model=model_name,
        binary_judgment=judgment,
        raw_output=raw_output,
    )


def _resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    normalized = dtype_name.lower()
    if normalized in {"auto", ""}:
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    attr = aliases.get(normalized)
    if not attr or not hasattr(torch, attr):
        raise ValueError(f"unsupported torch dtype: {dtype_name}")
    return getattr(torch, attr)


def _first_model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def _profile_is_binary_guard(profile: dict[str, Any]) -> bool:
    contract = profile.get("output_contract")
    if isinstance(contract, dict) and contract.get("type") == "binary_judgment":
        return True
    return str(profile.get("task") or "") == "agentdog_lite_binary"


def _format_binary_step(step: TrajectoryStep) -> str:
    role = step.role.upper()
    if step.type == "tool_call":
        segments = ["[AGENT]:"]
        if step.content:
            segments.append(step.content)
        action = step.tool_name or "tool.call"
        if step.arguments:
            action += _compact_json(step.arguments)
        segments.append(f"[ACTION]: {action}")
        return "\n".join(segments)
    if step.type == "observation" or step.role in {"tool", "environment"}:
        segments = ["[ENVIRONMENT]:"]
        if step.tool_name:
            segments.append(f"[TOOL]: {step.tool_name}")
        if step.content:
            segments.append(step.content)
        if step.arguments:
            segments.append(_compact_json(step.arguments))
        return "\n".join(segments)
    content = step.content if step.content is not None else ""
    if step.tool_name:
        content = f"{content}\n[TOOL]: {step.tool_name}".strip()
    if step.arguments:
        content = f"{content}\n[ARGS]: {_compact_json(step.arguments)}".strip()
    return f"[{role}]: {content}"


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _resolve_model_reference(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    project_path = ROOT / path
    if project_path.exists():
        return str(project_path)
    if path.exists():
        return str(path.resolve())
    return value
