"""Optional local Hugging Face model adapter.

This module is intentionally dependency-light at import time. Transformers,
torch, and PEFT-related packages are imported only when the local adapter is
instantiated on a GPU-capable environment.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from traceguard.chat_format import apply_chat_template
from traceguard.json_utils import extract_json_object
from traceguard.judge import ModelAdapter
from traceguard.model_profiles import profile_model_id, resolve_model_profile
from traceguard.prompts import build_remote_messages
from traceguard.schema import CostStats, RiskReport, TrajectoryCase


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
        self.model_name_or_path = (
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


def build_local_judge(
    *,
    model_profile: Optional[str] = None,
    model_path: Optional[str] = None,
    profile_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> LocalTransformersAdapter:
    return LocalTransformersAdapter(
        model_profile=model_profile,
        model_name_or_path=model_path,
        profile_path=profile_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
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
