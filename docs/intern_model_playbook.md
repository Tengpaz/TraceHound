# Intern Model Playbook

TraceHound now treats Intern models as first-class contest candidates while keeping the runtime adapter model-agnostic.

## Priority

1. `internlm/internlm3-8b-instruct`: first-choice local inference and LoRA/SFT candidate.
2. `internlm/internlm2_5-7b-chat`: fallback formal LoRA/SFT candidate.
3. `intern-s2-preview`: API candidate through the OpenAI-compatible adapter.
4. `internlm/internlm2_5-1_8b-chat`: low-cost smoke test before downloading larger checkpoints.
5. `internlm/internlm2_5-20b-chat`: larger fallback only if GPU memory permits.

Model profiles live in `configs/model_profiles.json`.

```bash
python scripts/list_model_profiles.py
python scripts/list_model_profiles.py --profile internlm3-8b-instruct
```

## Prompt Policy

Do not hardcode Qwen, Llama, or Intern-specific prompt strings.

- Local Hugging Face inference and SFT use `tokenizer.apply_chat_template`.
- If the tokenizer has no usable chat template, TraceHound falls back to a simple `SYSTEM / USER / ASSISTANT` prompt.
- Evaluation should still prefer `compressed` or `layered` mode. Long-context support does not remove the contest cost objective.

## API Validation

Copy `.env.server.example` to `.env` and fill the key:

```bash
TRACEHOUND_API_BASE=https://chat.intern-ai.org.cn/api/v1
TRACEHOUND_API_KEY=<your-key>
TRACEHOUND_MODEL=intern-s2-preview
TRACEHOUND_API_PATH=/chat/completions
```

Run a tiny smoke first:

```bash
python scripts/validate_api.py --data data/synthetic_eval.jsonl --limit 1 --judge api
python scripts/evaluate.py data/synthetic_eval.jsonl --judge hybrid --mode layered --limit 4
```

## Local Inference Smoke

Start with the 1.8B profile on a new GPU server:

```bash
python scripts/evaluate.py data/synthetic_eval.jsonl \
  --judge local \
  --model-profile internlm2_5-1_8b-chat \
  --mode compressed \
  --limit 1
```

Then validate the first-choice or fallback model:

```bash
python scripts/evaluate.py data/synthetic_eval.jsonl \
  --judge local \
  --model-profile internlm3-8b-instruct \
  --mode compressed \
  --limit 1
```

Use `--model-path /path/to/downloaded/checkpoint` when the contest server provides weights locally.

## SFT Plan

Default command only writes a plan and checks data:

```bash
python scripts/train_sft.py \
  --data data/synthetic_sft.jsonl \
  --model-profile internlm2_5-1_8b-chat \
  --output-dir checkpoints/intern-smoke-sft \
  --max-samples 32
```

On the GPU server, add `--run` to launch LoRA SFT:

```bash
python scripts/train_sft.py \
  --data data/synthetic_sft.jsonl \
  --model-profile internlm3-8b-instruct \
  --output-dir checkpoints/internlm3-tracehound-sft \
  --max-seq-length 4096 \
  --run
```

For the fallback formal run:

```bash
python scripts/train_sft.py \
  --data data/synthetic_sft.jsonl \
  --model-profile internlm2_5-7b-chat \
  --output-dir checkpoints/internlm25-7b-tracehound-sft \
  --max-seq-length 4096 \
  --run
```

## Preference / RL Plan

Default command writes `preference_plan.json`:

```bash
python scripts/train_preference.py \
  --data data/synthetic_preference.jsonl \
  --base-model checkpoints/internlm3-tracehound-sft \
  --model-profile internlm3-8b-instruct \
  --algorithm dpo \
  --output-dir checkpoints/internlm3-tracehound-dpo
```

On the GPU server, add `--run` for DPO or ORPO:

```bash
python scripts/train_preference.py \
  --data data/synthetic_preference.jsonl \
  --base-model checkpoints/internlm3-tracehound-sft \
  --model-profile internlm3-8b-instruct \
  --algorithm dpo \
  --output-dir checkpoints/internlm3-tracehound-dpo \
  --run
```

`grpo` is intentionally a reward-hook plan until the official scoring interface is known.

## Cost Strategy

Keep these runtime defaults even for 128K/256K-capable models:

- Rules early exit for obvious unsafe/safe cases.
- Risk span compression before model calls.
- Fallback to full context only for failures or targeted audits.
- API smoke limits first, then scale only after cost is acceptable.
