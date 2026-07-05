# TraceHound

TraceHound 是面向 Agent 轨迹安全评估与防御的项目。项目围绕 AgentDoG / ATBench / R-Judge 风格的轨迹级安全任务，完成了数据清洗、训练数据构建、二分类 guard 模型、三因素细粒度分类、六标签 + Reason 推理、GRPO 强化学习优化入口、在线 guardrail Demo、Project Page 和宣传海报。

当前仓库既保留可本地运行的 TraceHound Demo，也整合了外部 AgentDoG SFT/LoRA/GRPO 实验代码和关键 checkpoint，便于后续路演、复现实验、继续训练和打包交付。

## 项目产物

| 产物 | 路径 | 说明 |
|---|---|---|
| Project Page 主页 | `web_demo/index.html` | 类似 PPT 的一页一页滚动展示形式，可直接作为项目主页 |
| Web Demo | `scripts/serve_demo.py` | FastAPI + 原生 HTML/CSS/JS，支持轨迹评估、批量评测、guard model 配置、报告下载 |
| 16:9 路演海报 | `web_demo/assets/tracehound-project-poster-16x9-final.jpg` | 可作为路演宣传卡片和 Demo 视频封面 |
| A4 详细海报 | `web_demo/assets/tracehound-project-poster-a4.png`、`output/pdf/tracehound-project-poster-a4.pdf` | 用于宣传和较详细项目介绍 |
| 二分类训练数据 | `data/release/AgentDoG-Lite-TrainningDataset-Binary/` | AgentDoG-Lite 二分类 JSON / JSONL 训练包 |
| 细粒度 + Reason 数据 | `data/release/AgentDoG1.0-FineGrained-Unsafe-SourceReason-1000/` | 三因素细粒度分类和 Source/Reason 数据 |
| 导入 SFT 项目 | `external/agentdog_sft_project/` | 六标签 SFT、GRPO、ATBench/R-Judge 评测脚本 |
| 导入安全仓库 | `external/agentdog_safety_repo/` | 二分类 SFT、taxonomy LoRA、应用 guardrail 检查脚本 |
| 导入 6-label checkpoint | `checkpoints/imported/qwen35-0.8b-6label/` | Qwen3.5-0.8B 六标签全参数 SFT checkpoint，本地保留不进 Git |
| Hugging Face checkpoints | `https://huggingface.co/Tengpaz` | 关键实验 checkpoint 已上传到 Hugging Face，链接见下文 |

导入资产的来源、校验值和结果清单见 `docs/imported_agentdog_assets.md`。

## 快速启动

创建环境：

```bash
conda env create -f environment.yml
conda activate tracehound
```

如果环境已经存在：

```bash
conda env update -n tracehound -f environment.yml
conda activate tracehound
```

启动 Project Page 和 Demo：

```bash
python scripts/serve_demo.py --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

列出本地/API 模型配置：

```bash
python scripts/list_model_profiles.py
```

查看导入的 AgentDoG 工作流：

```bash
python scripts/imported_agentdog_workflows.py list
```

## 任务定义

项目关注四类轨迹来源和多个标签维度。

二分类任务：

| 标签 | 数值 | 含义 |
|---|---:|---|
| Safe | 0 | 轨迹安全，或存在风险但模型正确防御 |
| Unsafe | 1 | 轨迹存在风险且模型未采取有效防御 |

评测数据包含：

| 字段 | 说明 |
|---|---|
| `Label` | 二分类安全标签，Safe/Unsafe 分别映射为 0/1 |
| 三因素细粒度标签 | Risk Source、Failure Mode、Harm Type |
| `Reason` | 对模型分类理由的自然语言分析 |
| `Source` | 轨迹风险来源，包括 Safe、Unsafe、Benign、False_Refusal |

`Source` 的语义：

| Source | 说明 |
|---|---|
| Safe | 轨迹有风险，但是模型正确防御 |
| Unsafe | 轨迹有风险，且模型未采取任何防御手段 |
| Benign | 轨迹本身无风险 |
| False_Refusal | 轨迹有风险且模型采取了防御措施，但未成功防御，造成风险执行 |

六标签 + Reason 任务要求模型输出严格 JSON，核心字段包括：

```json
{
  "judgment": "safe|unsafe",
  "source": "Safe|Unsafe|Benign|False_Refusal",
  "risk_source": "...",
  "failure_mode": "...",
  "harm_type": "...",
  "reason": "..."
}
```

## 数据处理方法

初始提供的 AgentDoG1.0 训练数据只包括两部分：

- 基础二分类任务训练数据。
- 三因素细粒度分类任务训练数据。

项目在构建三因素细粒度分类任务、三因素细粒度分类 + Reason 分析任务时，对原始数据做了补充标注、筛选、清洗和合并。

处理策略：

1. 优先保留官方提供训练数据。我们认为这部分数据质量相对较高，因此避免让 LLM 生成标注过度覆盖原始高质量标签。
2. 对 Safe 轨迹补充三因素标签和 Reason。Safe 轨迹的三因素判断相对简单，LLM 补标风险低于 Unsafe。
3. 对三因素细粒度数据中的 Unsafe 轨迹补充 Reason。Unsafe 轨迹标签更复杂，因此保留已有三因素标注，只新增解释字段。
4. 抛弃二分类训练数据中的 Unsafe 轨迹，直接合并三因素细粒度分类数据中的 Unsafe 轨迹，降低低质量自动标注带来的噪声。
5. 合并后统一输出 JSON 格式，保证训练和评测阶段的输出约束与赛题一致。

数据清洗问题：

| 问题 | 处理方式 |
|---|---|
| Prompt 对输出约束不一致 | 统一修正为赛题要求的 JSON 输出格式 |
| 三因素标签名格式变动 | 统一大小写、间隔符和 canonical label |
| 二分类数据存在约 2000 条重复 | 去重后再构建训练数据 |
| LLM Reason 可能引入噪声 | 只在需要补齐的字段上使用 LLM，并保留原始高质量标签 |
| 训练/评测污染风险 | 评测集默认只用于 schema 和标签对齐，不混入训练输出 |

关键本地数据：

```text
data/release/AgentDoG-Lite-TrainningDataset-Binary/
data/release/AgentDoG1.0-FineGrained-Unsafe-SourceReason-1000/
external/agentdog_sft_project/outputs/data/agentdog_binary_sft.jsonl
external/agentdog_sft_project/outputs/data/agentdog_6label_sft.jsonl
```

## 方法介绍

### TraceHound 基线

TraceHound 的轻量 baseline 包含：

- 轨迹 schema 解析和标准化。
- 风险 span 压缩。
- 规则预筛。
- 启发式安全判断。
- 本地/API judge 适配。
- 输出格式校验。
- 成本统计和评测报告。
- Runtime guardrail endpoint。
- Claude Code、Codex-style wrapper、OpenClaw、通用 Agent middleware 示例。

### 二分类 SFT

项目使用官方训练数据对 Qwen3.5-0.8B 做任务特定微调，目标是让模型稳定输出：

```json
{"judgment":"safe"}
```

或：

```json
{"judgment":"unsafe"}
```

对比方案：

- Qwen3.5-0.8B 基座模型。
- TraceHound-Basev1，全参数 SFT。
- TraceHound-LoRAv1，LoRA 微调。

### 三因素细粒度 SFT

为了定位轨迹危险因素，项目训练了三因素细粒度分类模型，输出 Risk Source、Failure Mode、Harm Type，并评估 ATBench 与 R-Judge。

### 六标签 + Reason SFT

在三因素基础上，进一步加入：

- `judgment`
- `source`
- `risk_source`
- `failure_mode`
- `harm_type`
- `reason`

导入的关键 checkpoint：

```text
checkpoints/imported/qwen35-0.8b-6label/
```

对应模型配置：

```text
agentdog-qwen3_5-0_8b-6label-imported
```

### GRPO 强化学习优化

项目也对三因素 + Reason 推理任务接入了 GRPO 强化学习训练优化路线。相关代码和配置已经整合：

```text
external/agentdog_sft_project/scripts/train_6label_grpo.py
external/agentdog_sft_project/configs/grpo_6label_defaults.json
```

当前 GRPO 的主要作用是继续优化六标签 JSON 输出、分类字段匹配和 Reason 合理性。由于该方向的新实验结果尚未最终产出，本文档只保留方法和运行入口，不把未定稿数据作为正式主结果。

## 实验结果

### 二分类结果

ATBench 指标：

| 模型 | Accuracy | F1 | Invalid Rate | Output Tokens mean/max/min/median |
|---|---:|---:|---:|---|
| Qwen3.5-0.8B | 0.5185 | 0.6829 | 0.0000 | 13 / 13 / 13 / 13 |
| TraceHound-Basev1 | 0.7303 | 0.7791 | 0.0000 | 13 / 13 / 13 / 13 |
| TraceHound-LoRAv1 | 0.6767 | 0.7188 | 0.0067 | 13.12 / 32 / 13 / 13 |

导入 GitHub 实验仓库中的二分类 SFT 结果：

| 数据集 | Accuracy | F1 | Precision Unsafe | Recall Unsafe | Invalid Rate |
|---|---:|---:|---:|---:|---:|
| ATBench / summer-camp | 0.7303 | 0.7791 | 0.6771 | 0.9174 | 0.0000 |
| R-Judge | 0.8103 | 0.8315 | 0.7834 | 0.8859 | 0.0000 |

结论：

- 全参数 SFT 相比基座模型显著提升二分类准确率和 F1。
- LoRA 有提升，但不如全参数 SFT，说明小模型在该安全分类任务上需要更充分的参数对齐。
- 二分类 SFT 输出格式稳定，主结果达到 100% 有效输出。

### 三因素细粒度分类结果

ATBench 指标：

| 模型 | Accuracy | F1 | Invalid Rate | Output Tokens mean/max/min/median |
|---|---:|---:|---:|---|
| Qwen3.5-0.8B | 0.1367 | 0.2897 | 0.7200 | - |
| AgentDoG1.0-Qwen3.5-0.8B | 0.6433 | 0.7116 | 0.0000 | 34.79 / 40 / 28 / 34.5 |

R-Judge 指标：

| 模型 | Accuracy | F1 | Invalid Rate | Output Tokens mean/max/min/median |
|---|---:|---:|---:|---|
| Qwen3.5-0.8B | 0.2571 | 0.6259 | 0.7074 | - |
| AgentDoG1.0-Qwen3.5-0.8B | 0.6259 | 0.4903 | 0.0018 | 30.86 / 39 / 20 / 30 |

ATBench 三因素拆分：

| 维度 | Baseline Acc | SFT Acc | Baseline Macro-F1 | SFT Macro-F1 |
|---|---:|---:|---:|---:|
| Risk Source | 0.0267 | 0.1833 | 0.0336 | 0.1520 |
| Failure Mode | 0.0133 | 0.2500 | 0.0125 | 0.0642 |
| Harm Type | 0.0567 | 0.2867 | 0.0638 | 0.2434 |

结论：

- 微调对准确率和格式规范性提升明显，尤其 Invalid Rate 从高比例无效输出降到接近 0。
- R-Judge 上出现 F1 下降现象，主要是因为基座模型无效率高、偏向 Unsafe，导致计算基数和召回率结构异常，使 F1 指标被动占优。
- 从有效输出和真实分类能力看，SFT 模型更稳定、更适合作为下游 guardrail 组件。

### 六标签 + Reason 结果与资产

导入的六标签 checkpoint 保留在：

```text
checkpoints/imported/qwen35-0.8b-6label/
```

导入历史评测目录中保留了 R-Judge 和训练集 pass@k 结果：

| 任务 | 数据集 | Accuracy | Macro-F1 / F1 | 有效输出 |
|---|---|---:|---:|---:|
| 六标签二分类字段 | R-Judge | 0.8582 | 0.8581 | 100% |
| 六标签训练集 pass@8 | Train set | 1.0000 | 1.0000 | 100% 有效 rollout |

这些结果用于说明六标签模型在严格 JSON 输出和二分类字段上具备稳定性。正式论文主表仍以已确认的二分类 SFT、LoRA 对比和三因素细粒度 SFT 为核心。

## 复现实验命令

### 构建 AgentDoG-Lite 二分类训练数据

```bash
python scripts/build_agentdog_lite_binary.py \
  --out data/release/AgentDoG-Lite-TrainningDataset-Binary
```

### 构建 FineGrained + Source/Reason 数据

```bash
python scripts/build_agentdog_finegrained_source_reason.py
```

### 评估本地二分类 guard

```bash
python scripts/evaluate_lite_binary_model.py --datasets atbench,rjudge
```

### 使用导入的六标签 checkpoint 评测

先查看命令：

```bash
python scripts/imported_agentdog_workflows.py eval-6label-atbench --dry-run
python scripts/imported_agentdog_workflows.py eval-6label-rjudge --dry-run
```

实际运行：

```bash
python scripts/imported_agentdog_workflows.py eval-6label-atbench
python scripts/imported_agentdog_workflows.py eval-6label-rjudge
```

### 运行六标签 SFT / GRPO

```bash
python scripts/imported_agentdog_workflows.py train-6label-sft --dry-run
python scripts/imported_agentdog_workflows.py train-6label-grpo --dry-run
```

去掉 `--dry-run` 后会执行外部项目中的 `torchrun` 命令。运行前需要确认 GPU、CUDA、DeepSpeed、Transformers、TRL 等训练依赖可用。

### 运行导入的二分类 SFT / taxonomy LoRA

```bash
python scripts/imported_agentdog_workflows.py train-binary-lr8e6 --dry-run
python scripts/imported_agentdog_workflows.py train-taxonomy-lora --dry-run
```

相关原始配置在：

```text
external/agentdog_safety_repo/configs/binary_lr8e6_steps330.json
external/agentdog_safety_repo/configs/taxonomy_lora_lr8e6_steps200.json
```

## 目录结构

```text
TraceHound/
  configs/                         模型、数据生成和 AgentDoG 配置
  data/                            训练数据、release 数据、smoke 数据
  docs/                            复现、部署、导入资产和设计文档
  external/
    agentdog_sft_project/          导入的 SFT/6-label/GRPO 项目
    agentdog_safety_repo/          导入的二分类 SFT 与 LoRA 实验仓库
  checkpoints/imported/            本地保留的大 checkpoint，不提交 Git
  models/                          本地 guard 模型，不提交 Git
  output/pdf/                      海报 PDF 输出
  scripts/                         TraceHound 主脚本和导入工作流入口
  tests/                           单元测试
  traceguard/                      TraceHound 核心 Python 包
  web_demo/                        Project Page、Demo 前端、海报资源
```

## 本地模型与 checkpoint

`.gitignore` 会忽略以下大文件目录：

```text
checkpoints/
models/
outputs/
runs/
*.safetensors
*.pt
*.pth
```

因此本地关键 checkpoint 会保留在机器和最终 zip 包中，但不会被 Git 提交。

当前已保留：

```text
models/TraceHound-Base-Qwen3.5-0.8B-Binary/
checkpoints/imported/qwen35-0.8b-6label/
```

六标签 checkpoint 的主权重：

```text
checkpoints/imported/qwen35-0.8b-6label/model.safetensors
```

SHA-256：

```text
1e6beee966eddb65f10c1ade560481d3fbe3978d8a70f03dbddd071166c6f0d6
```

## Hugging Face Checkpoints

关键实验 checkpoint 已上传到 Hugging Face，便于在新机器上直接下载复现实验。`models/Qwen3.5-0.8B/` 是基座模型本地缓存，不作为 TraceHound 实验 checkpoint 重复上传；实验模型均基于 `Qwen/Qwen3.5-0.8B`。

| Checkpoint | Hugging Face | 本地对应路径 | 说明 |
|---|---|---|---|
| TraceHound 二分类 full-SFT | [Tengpaz/TraceHound-Base-Qwen3.5-0.8B-Binary](https://huggingface.co/Tengpaz/TraceHound-Base-Qwen3.5-0.8B-Binary) | `models/TraceHound-Base-Qwen3.5-0.8B-Binary/` | 二分类 guard 主模型，ATBench Accuracy 0.7303 / F1 0.7791 |
| AgentDoG 六标签 full-SFT | [Tengpaz/AgentDoG-Qwen3.5-0.8B-6Label](https://huggingface.co/Tengpaz/AgentDoG-Qwen3.5-0.8B-6Label) | `checkpoints/imported/qwen35-0.8b-6label/` | 六标签 + Reason SFT checkpoint |
| TraceHound unified full-SFT | [Tengpaz/TraceHound-Qwen3.5-0.8B-Unified-Full-SFT](https://huggingface.co/Tengpaz/TraceHound-Qwen3.5-0.8B-Unified-Full-SFT) | `checkpoints/qwen3_5_0_8b_full_sft_unified_notrunc_20260705_000140/` | 保留 final、checkpoint-200、checkpoint-225、optimizer/scheduler/RNG/trainer state |
| TraceHound Lite Binary LoRA | [Tengpaz/TraceHound-Qwen3.5-0.8B-Lite-Binary-LoRA](https://huggingface.co/Tengpaz/TraceHound-Qwen3.5-0.8B-Lite-Binary-LoRA) | `checkpoints/qwen3_5_0_8b_lite_binary_preserve_target_150039/` | LoRA 对比实验 checkpoint，包含 checkpoint-150 |

下载示例：

```bash
huggingface-cli download Tengpaz/TraceHound-Base-Qwen3.5-0.8B-Binary \
  --local-dir models/TraceHound-Base-Qwen3.5-0.8B-Binary

huggingface-cli download Tengpaz/AgentDoG-Qwen3.5-0.8B-6Label \
  --local-dir checkpoints/imported/qwen35-0.8b-6label

huggingface-cli download Tengpaz/TraceHound-Qwen3.5-0.8B-Unified-Full-SFT \
  --local-dir checkpoints/qwen3_5_0_8b_full_sft_unified_notrunc_20260705_000140

huggingface-cli download Tengpaz/TraceHound-Qwen3.5-0.8B-Lite-Binary-LoRA \
  --local-dir checkpoints/qwen3_5_0_8b_lite_binary_preserve_target_150039
```

## 远程 GPU 服务器

无 Docker 的 Linux GPU 服务器推荐使用 conda bootstrap：

```bash
git clone git@github.com:Tengpaz/TraceHound.git
cd TraceHound
cp .env.server.example .env
bash scripts/bootstrap_remote.sh
```

如果服务器没有 conda/mamba/micromamba：

```bash
bash scripts/install_miniconda_linux.sh
export PATH="$HOME/miniconda3/bin:$PATH"
bash scripts/bootstrap_remote.sh
```

启动远程 demo：

```bash
conda activate tracehound-gpu
bash scripts/run_remote_demo.sh
```

Slurm 环境可使用：

```bash
bash scripts/slurm_gpu_test.sh
TRACEHOUND_SLURM_GPUS=1 bash scripts/slurm_run.sh 'python scripts/evaluate_lite_binary_model.py --datasets atbench'
```

详细说明见：

```text
docs/remote_gpu_deploy.md
docs/slurm_cluster_usage.md
docs/training_gpu.md
```

## 测试

快速检查新导入入口：

```bash
python -m py_compile scripts/imported_agentdog_workflows.py
python scripts/imported_agentdog_workflows.py list
```

运行核心测试：

```bash
pytest
```

如果本机没有 GPU 训练依赖，训练脚本不应在本地强行运行；可以先使用 `--dry-run` 检查命令和路径。

## 总结结论

1. 数据质量是主要瓶颈。项目通过去重、Prompt 修正、标签 canonicalize、Source/Reason 补标和高质量数据优先合并，降低了 LLM 自动标注噪声。
2. 二分类任务上，全参数 SFT 相比 Qwen3.5-0.8B 基座模型有显著提升，ATBench Accuracy 从 0.5185 提升到 0.7303，F1 从 0.6829 提升到 0.7791。
3. LoRA 微调有效但提升有限，TraceHound-LoRAv1 低于全参数 SFT，说明该安全判别任务对小模型的整体参数对齐要求较高。
4. 三因素细粒度分类中，SFT 明显改善准确率和输出格式稳定性，Invalid Rate 从高比例无效输出降低到接近 0。
5. R-Judge 上部分 F1 指标需要结合 Invalid Rate 和类别分布解释，不能只看单一 F1；基座模型由于无效率高和偏 Unsafe，可能在召回结构上获得虚高分数。
6. 六标签 + Reason 方向已经具备 SFT checkpoint、评测脚本和 GRPO 优化入口，后续任务是完成 GRPO 正式评测、加入 Reason 质量指标，并将最终结果更新到 Project Page 和论文表格。

## 后续任务

- 完成 GRPO 六标签 + Reason 的最终训练与评测。
- 为 Reason 增加独立质量指标，例如字段一致性、证据覆盖率和人工抽检。
- 将 Project Page 中的结果表与最终 README/论文主表保持同步。
- 增加更多下游 Agent runtime hook 的真实场景验证。
- 对比更大参数模型和蒸馏策略，评估成本、延迟、准确率之间的取舍。
