# 导入资产清单

本文档记录本次整合进 TraceHound 的外部代码、数据输出和 checkpoint。目标是保留所有关键实验能力，同时避免把大权重误提交到 Git。

## 来源

| 来源 | 本地导入位置 | 内容 |
|---|---|---|
| `/Users/a1234/Downloads/agentdog_sft_project.tar.gz` | `external/agentdog_sft_project/` | AgentDoG 二分类、三因素、六标签 + Reason SFT/GRPO 训练与评测项目 |
| `/Users/a1234/Downloads/qwen35-0.8b-6label.tar.gz` | `checkpoints/imported/qwen35-0.8b-6label/` | Qwen3.5-0.8B 六标签全参数 SFT checkpoint |
| `git@github.com:JOEY-311/agentdog-safety-repo.git` | `external/agentdog_safety_repo/` | 二分类 SFT、taxonomy LoRA、应用 guardrail 检查和 API 数据增强脚本 |

## 校验值

| 文件 | SHA-256 |
|---|---|
| `agentdog_sft_project.tar.gz` | `5d8d1b12580bbc4d129aac06268991b948ab5b0ca3eee8e9f8e5b1a115280c55` |
| `qwen35-0.8b-6label.tar.gz` | `d74241a7b91a8b0ec71726c163a7c6342d1fcc016ae8297e5002c536e75c404d` |
| `checkpoints/imported/qwen35-0.8b-6label/model.safetensors` | `1e6beee966eddb65f10c1ade560481d3fbe3978d8a70f03dbddd071166c6f0d6` |

## Checkpoint

`checkpoints/imported/qwen35-0.8b-6label/` 保留了关键六标签实验 checkpoint：

- `model.safetensors`
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `training_args.bin`
- `run_config.json`
- `runs/` TensorBoard 事件文件

该目录被 `.gitignore` 中的 `checkpoints/` 规则忽略，不会进入 Git 提交；但它会保留在本机，并会进入最终项目 zip 包。

对应模型配置已加入 `configs/model_profiles.json`：

```text
agentdog-qwen3_5-0_8b-6label-imported
```

## 导入代码

### `external/agentdog_sft_project/`

保留了六标签主线实验的完整代码和输出：

- `scripts/prepare_sft_data.py`
- `scripts/prepare_6label_sft_data.py`
- `scripts/train_sft.py`
- `scripts/train_6label_sft.py`
- `scripts/train_6label_grpo.py`
- `scripts/binary_safety_eval.py`
- `scripts/evaluate_atbench.py`
- `scripts/evaluate_rjudge.py`
- `scripts/trainset_passk_eval.py`
- `scripts/merge_eval_shards.py`
- `configs/training_defaults.json`
- `configs/training_6label_defaults.json`
- `configs/grpo_6label_defaults.json`
- `src/guardrail/prompts.py`
- `src/guardrail/taxonomy.py`
- `src/guardrail/metrics.py`
- `2026_summer_camp_teseset/`
- `outputs/data/agentdog_binary_sft.jsonl`
- `outputs/data/agentdog_6label_sft.jsonl`
- `outputs/eval/` 历史评测输出

其中 `outputs/` 属于本地实验产物，会被 Git 忽略，但已保留在工作区和最终 zip 包。

### `external/agentdog_safety_repo/`

保留了 GitHub 仓库里的二分类和 taxonomy LoRA 分支：

- `scripts/train/train_qwen_binary_sft_maxsteps.py`
- `scripts/train/train_qwen_taxonomy_lora.py`
- `scripts/eval/evaluate_binary_val.py`
- `scripts/eval/evaluate_unified_full_model.py`
- `scripts/eval/evaluate_unified_lora_model.py`
- `scripts/application/qwen_binary_verifier_email.py`
- `scripts/application/qwen_binary_verifier_database.py`
- `scripts/data_generation/augment_safe_samples_with_api.py`
- `configs/binary_lr8e6_steps330.json`
- `configs/taxonomy_lora_lr8e6_steps200.json`
- `model_params/`
- `reports/`

该仓库不包含大权重和原始数据，只有配置、小参数快照和结果摘要。

## 统一入口

新增脚本：

```bash
python scripts/imported_agentdog_workflows.py list
```

常用命令：

```bash
python scripts/imported_agentdog_workflows.py eval-6label-atbench --dry-run
python scripts/imported_agentdog_workflows.py eval-6label-rjudge --dry-run
python scripts/imported_agentdog_workflows.py train-6label-sft --dry-run
python scripts/imported_agentdog_workflows.py train-6label-grpo --dry-run
python scripts/imported_agentdog_workflows.py train-binary-lr8e6 --dry-run
python scripts/imported_agentdog_workflows.py train-taxonomy-lora --dry-run
```

实际运行 GPU 训练前，需要确认外部脚本里的 `model_path`、数据路径和 CUDA/DeepSpeed 环境符合当前机器。

## 已保留的关键结果

导入的二分类 SFT 报告给出了稳定且更适合宣传的主指标：

| 数据集 | Accuracy | F1 | Invalid Rate | 说明 |
|---|---:|---:|---:|---|
| ATBench / summer-camp | 0.7303 | 0.7791 | 0.0000 | `reports/binary_lr8e6_steps330/summer_camp_metrics.json` |
| R-Judge | 0.8103 | 0.8315 | 0.0000 | `reports/binary_lr8e6_steps330/rjudge_metrics.json` |

导入的六标签 + Reason 评测输出包含：

| 数据集 | Accuracy | Macro-F1 | 有效输出率 | 说明 |
|---|---:|---:|---:|---|
| R-Judge | 0.8582 | 0.8581 | 100% | `outputs/eval/rjudge-qwen35-0.8b-6label-grpo-exp100/metrics.json` |
| 六标签训练集 pass@8 | 1.0000 | 1.0000 | 100% 有效 rollout | `outputs/eval/qwen35-0.8b-6label-train-pass8-hf/merged/metrics.json` |

GRPO 训练代码和配置已保留在 `external/agentdog_sft_project/scripts/train_6label_grpo.py` 和 `external/agentdog_sft_project/configs/grpo_6label_defaults.json`。由于该方向的完整新实验数据尚未最终产出，README 和 Project Page 中只描述其作为六标签 + Reason 推理优化路线，不把未完成数据写成正式结果。
