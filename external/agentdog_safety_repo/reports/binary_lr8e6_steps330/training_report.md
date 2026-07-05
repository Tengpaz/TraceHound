# Qwen3.5-0.8B 二分类 SFT 训练报告

生成时间：2026-07-05  
实验名称：`binary_dedup_lr8e-6`  
模型：`Qwen3.5-0.8B`  
最终 checkpoint：`/root/autodl-tmp/sft_dedup_binary/outputs/models/binary_dedup_lr8e-6/checkpoint-450`

## 1. 实验目标

本次实验在去重后的二分类数据上使用较低学习率 `8e-6` 重新训练，用于和上一版 `1e-5` 做对比。

目标输出仍为严格 JSON：

```json
{
  "judgment": "safe"
}
```

或：

```json
{
  "judgment": "unsafe"
}
```

## 2. 数据设置

训练数据与上一版去重实验一致：

| split | total | safe | unsafe |
|---|---:|---:|---:|
| train | 1800 | 900 | 900 |
| validation | 200 | 100 | 100 |
| total | 2000 | 1000 | 1000 |

去重检查：

- 原始样本数：`4000`
- 唯一 prompt 数：`2000`
- train/validation prompt overlap：`0`

服务器数据路径：

- train：`/root/autodl-tmp/sft_dedup_binary/data/agentdog_binary_dedup_train.jsonl`
- validation：`/root/autodl-tmp/sft_dedup_binary/data/agentdog_binary_dedup_val.jsonl`

## 3. 训练参数

| 参数 | 值 |
|---|---:|
| learning rate | `8e-6` |
| max length | `8192` |
| per-device train batch size | `1` |
| gradient accumulation steps | `8` |
| effective batch size | `8` |
| train examples | `1800` |
| epochs | `2` |
| total optimizer steps | `450` |
| warmup ratio | `0.03` |
| warmup steps | `14` |
| scheduler | cosine |
| precision | bf16 |
| gradient checkpointing | enabled |
| save steps | `25` |
| save total limit | `1` |
| seed | `42` |

## 4. 训练结果

训练已完成：

- final step：`450 / 450`
- final loss：`1.361089580314001e-05`
- final learning rate：`0.0`
- final checkpoint：`checkpoint-450`
- checkpoint 保留数量：`1`
- 训练耗时：约 `110.2` 分钟

Loss 曲线：

![loss curve](E:/summercamp/SAIL/training_plots/dedup_lr8e-6/loss_curve_dedup_lr8e-6.svg)

Loss 关键点：

| item | step | loss |
|---|---:|---:|
| max | 10 | `0.3642228988697752` |
| min | 330 | `1.2861720313139812e-05` |
| final | 450 | `1.361089580314001e-05` |

本地文件：

- loss 图：`E:\summercamp\SAIL\training_plots\dedup_lr8e-6\loss_curve_dedup_lr8e-6.svg`
- loss CSV：`E:\summercamp\SAIL\training_plots\dedup_lr8e-6\loss_curve_dedup_lr8e-6.csv`
- 训练日志：`E:\summercamp\SAIL\training_plots\dedup_lr8e-6\train.log`

## 5. Clean Validation

Clean validation 使用去重后内部 validation set，共 `200` 条，safe/unsafe 各 `100` 条。

最终 checkpoint-450 的 clean validation：

| metric | value |
|---|---:|
| accuracy | `1.0` |
| F1 unsafe | `1.0` |
| precision unsafe | `1.0` |
| recall unsafe | `1.0` |
| invalid rate | `0.0` |
| TP | `100` |
| FP | `0` |
| TN | `100` |
| FN | `0` |
| output tokens mean | `15.0` |
| output tokens max | `15` |
| output tokens min | `15` |
| output tokens median | `15.0` |

Validation 曲线记录：

- 本地：`E:\summercamp\SAIL\training_reports\dedup_lr8e-6\val_history.csv`
- 服务器：`/root/autodl-tmp/sft_dedup_binary/outputs/val_eval/binary_dedup_lr8e-6/val_history.csv`

## 6. 外部评测一：Summer Camp Testset

评测口径：

- 使用之前相同的二分类 testset，共 `864` 条
- 使用同一版 binary prompt
- `enable_thinking=False`
- 严格 JSON 判定

结果：

| metric | value |
|---|---:|
| total | `864` |
| invalid rate | `0.0` |
| accuracy | `0.7152777777777778` |
| F1 score | `0.7515151515151517` |
| precision unsafe | `0.6863468634686347` |
| recall unsafe | `0.8303571428571429` |
| think tag rate | `0.0` |

混淆矩阵：

| item | count |
|---|---:|
| TP | `372` |
| FP | `170` |
| TN | `246` |
| FN | `76` |

Output Token Cost：

| statistic | value |
|---|---:|
| mean | `15.0` |
| max | `15` |
| min | `15` |
| median | `15.0` |

输出路径：

- 本地：`E:\summercamp\SAIL\eval_outputs\qwen35_08b_sft_dedup_lr8e-6_v4_exact_prompt`
- 服务器：`/root/autodl-tmp/eval_qwen35_08b_sft_dedup_lr8e-6_v4_exact_prompt`

## 7. 外部评测二：Summer Camp R-Judge

数据：

- `summer_camp_rjudge.json`
- total：`564`
- safe：`266`
- unsafe：`298`

结果：

| metric | value |
|---|---:|
| total | `564` |
| invalid rate | `0.0` |
| accuracy | `0.799645390070922` |
| F1 unsafe | `0.8061749571183534` |
| precision unsafe | `0.8245614035087719` |
| recall unsafe | `0.7885906040268457` |
| F1 safe | `0.7926605504587156` |
| precision safe | `0.7741935483870968` |
| recall safe | `0.8120300751879699` |
| think tag rate | `0.0` |

混淆矩阵：

| item | count |
|---|---:|
| TP | `235` |
| FP | `50` |
| TN | `216` |
| FN | `63` |

Output Token Cost：

| statistic | value |
|---|---:|
| mean | `15.0` |
| max | `15` |
| min | `15` |
| median | `15.0` |

输出路径：

- 本地：`E:\summercamp\SAIL\eval_outputs\rjudge_sft_dedup_lr8e-6_original_prompt`
- 服务器：`/root/autodl-tmp/eval_rjudge_sft_dedup_lr8e-6_original_prompt`

## 8. 与 1e-5 对比

| Run | Summer Accuracy | Summer F1 | R-Judge Accuracy | R-Judge F1 |
|---|---:|---:|---:|---:|
| dedup lr=1e-5 | `0.6968` | `0.7305` | `0.7926` | `0.7937` |
| dedup lr=8e-6 | `0.7153` | `0.7515` | `0.7996` | `0.8062` |

`8e-6` 在两个外部评测集上都略优于 `1e-5`：

- Summer F1：`0.7305 -> 0.7515`
- R-Judge F1：`0.7937 -> 0.8062`

## 9. 结论

本次 `8e-6` 训练完成且稳定，内部 clean validation 仍为满分。更重要的是，两个外部评测也比 `1e-5` 版本略有提升。

当前推荐 checkpoint：

`/root/autodl-tmp/sft_dedup_binary/outputs/models/binary_dedup_lr8e-6/checkpoint-450`

推荐理由：

- Summer camp testset F1 更高
- R-Judge F1 更高
- 输出格式稳定，invalid rate 为 `0.0`
- 没有 `<think>` 输出

后续建议：

1. 若继续调参，可尝试 `5e-6`，看是否进一步降低 FP。
2. 对 `8e-6` 的 false positives / false negatives 做 error analysis。
3. 如果硬盘紧张，可继续只保留 `1e-5` 和 `8e-6` 两个去重版最终 checkpoint。
