import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_judgment(text):
    text = text.strip()
    try:
        obj = json.loads(text)
        if set(obj.keys()) == {"judgment"} and obj["judgment"] in {"safe", "unsafe"}:
            return obj["judgment"], True
    except Exception:
        pass
    match = re.search(r'\{[^{}]*"judgment"\s*:\s*"(safe|unsafe)"[^{}]*\}', text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if set(obj.keys()) == {"judgment"} and obj["judgment"] in {"safe", "unsafe"}:
                return obj["judgment"], text == match.group(0)
        except Exception:
            pass
    return None, False


def target_judgment(row):
    content = row["messages"][1]["content"].strip()
    obj = json.loads(content)
    return obj["judgment"]


def encode_prompt(tokenizer, prompt):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    return encoded


def metrics_from(rows):
    tp = fp = tn = fn = 0
    pred_safe = pred_unsafe = target_safe = target_unsafe = invalid = 0
    for row in rows:
        target = row["target"]
        pred = row["parsed_judgment"] if row["valid_exact_json"] else None
        if target == "unsafe":
            target_unsafe += 1
        else:
            target_safe += 1
        if pred is None:
            invalid += 1
            pred = "safe"
        if pred == "unsafe":
            pred_unsafe += 1
        else:
            pred_safe += 1
        if pred == "unsafe" and target == "unsafe":
            tp += 1
        elif pred == "unsafe" and target == "safe":
            fp += 1
        elif pred == "safe" and target == "safe":
            tn += 1
        elif pred == "safe" and target == "unsafe":
            fn += 1
    total = len(rows)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "total": total,
        "accuracy": accuracy,
        "precision_unsafe": precision,
        "recall_unsafe": recall,
        "f1_unsafe": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pred_safe": pred_safe,
        "pred_unsafe": pred_unsafe,
        "target_safe": target_safe,
        "target_unsafe": target_unsafe,
        "invalid_outputs": invalid,
        "invalid_rate": invalid / total if total else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.validation_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    predictions = []
    token_counts = []
    for idx, row in enumerate(rows, 1):
        prompt = row["messages"][0]["content"]
        target = target_judgment(row)
        encoded = encode_prompt(tokenizer, prompt)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0][encoded["input_ids"].shape[-1]:]
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        parsed, valid = parse_judgment(raw_output)
        token_counts.append(int(output_ids.shape[-1]))
        predictions.append({
            "id": row.get("id"),
            "target": target,
            "raw_output": raw_output,
            "parsed_judgment": parsed,
            "valid_exact_json": valid,
            "output_tokens": int(output_ids.shape[-1]),
        })
        if idx % 50 == 0:
            print(f"validated {idx}/{len(rows)}", flush=True)

    metrics = metrics_from(predictions)
    metrics["checkpoint"] = Path(args.model_path).name
    metrics["model_path"] = args.model_path
    metrics["output_token_cost"] = {
        "mean": sum(token_counts) / len(token_counts),
        "max": max(token_counts),
        "min": min(token_counts),
        "median": statistics.median(token_counts),
    }

    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
