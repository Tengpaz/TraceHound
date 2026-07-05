import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class SftDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def chat_input_ids(self, messages):
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors=None,
        )
        if not isinstance(encoded, list):
            return encoded["input_ids"]
        return encoded

    def __getitem__(self, idx):
        row = self.rows[idx]
        user = row["messages"][0]["content"]
        assistant = row["messages"][1]["content"].strip()
        prompt_ids = self.chat_input_ids([{"role": "user", "content": user}])
        target_ids = self.tokenizer.encode(assistant + self.tokenizer.eos_token, add_special_tokens=False)
        if len(prompt_ids) + len(target_ids) > self.max_length:
            keep_prompt = max(1, self.max_length - len(target_ids))
            prompt_ids = prompt_ids[:keep_prompt]
        input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
        labels = torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_token_id):
    input_ids = pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([x["labels"] for x in batch], batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def save_checkpoint(model, tokenizer, output_dir, step, save_total_limit):
    ckpt = output_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt, safe_serialization=True)
    tokenizer.save_pretrained(ckpt)
    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    while len(checkpoints) > save_total_limit:
        victim = checkpoints.pop(0)
        shutil.rmtree(victim, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3.5-0.8B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = SftDataset(args.train_file, tokenizer, args.max_length)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    planned_steps = steps_per_epoch * args.num_train_epochs
    total_steps = args.max_steps if args.max_steps and args.max_steps > 0 else planned_steps
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(json.dumps({
        "train_examples": len(train_data),
        "steps_per_epoch": steps_per_epoch,
        "planned_steps": planned_steps,
        "total_steps": total_steps,
        "max_steps": args.max_steps,
        "warmup_steps": warmup_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "learning_rate": args.learning_rate,
    }, indent=2), flush=True)

    global_step = 0
    running_loss = 0.0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.num_train_epochs):
        for batch_idx, batch in enumerate(loader, 1):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

            if batch_idx % args.gradient_accumulation_steps == 0 or batch_idx == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0 or global_step == 1:
                    avg_loss = running_loss / max(1, args.logging_steps)
                    running_loss = 0.0
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - started
                    print(
                        json.dumps({
                            "step": global_step,
                            "total_steps": total_steps,
                            "epoch": epoch + 1,
                            "loss": avg_loss,
                            "lr": lr,
                            "elapsed_sec": round(elapsed, 1),
                        }),
                        flush=True,
                    )

                if global_step % args.save_steps == 0 or global_step == total_steps:
                    print(f"Saving checkpoint-{global_step}", flush=True)
                    save_checkpoint(model, tokenizer, output_dir, global_step, args.save_total_limit)

                if global_step >= total_steps:
                    print("Training complete", flush=True)
                    return

    print("Training complete", flush=True)


if __name__ == "__main__":
    main()
