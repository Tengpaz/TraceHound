const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const caseSelect = $("#caseSelect");
const modeSelect = $("#modeSelect");
const judgeSelect = $("#judgeSelect");
const caseInput = $("#caseInput");
const runBtn = $("#runBtn");
const statusEl = $("#status");
const langToggle = $("#langToggle");
const jsonUpload = $("#jsonUpload");
const dropZone = $("#dropZone");
const batchRunBtn = $("#batchRunBtn");
const evalDatasetSelect = $("#evalDatasetSelect");
const refreshDatasetsBtn = $("#refreshDatasetsBtn");

let currentCase = null;
let runtimeInfo = null;
let modelInfo = null;
let enchantmentInfo = null;
let uploadedCases = [];
let evalDatasets = [];
let language = "zh";

const scenarioLabels = {
  zh: {
    shell: "Shell 命令",
    file: "文件",
    browser: "浏览器",
    email: "邮件",
    database: "数据库",
    code_executor: "代码执行器",
    calendar: "日历",
    credential: "凭据",
  },
  en: {},
};

const copy = {
  zh: {
    "common.checkingApi": "检查 API",
    "common.loading": "加载中",
    "common.idle": "空闲",
    "common.ready": "就绪",
    "common.running": "运行中",
    "common.done": "完成",
    "common.error": "错误",
    "common.available": "可用",
    "common.notAvailable": "不可用",
    "common.notConfigured": "未配置",
    "common.model": "模型",
    "common.noData": "无数据",
    "common.apiReady": "API 已就绪",
    "common.apiMissingKey": "API 缺少密钥",
    "common.apiNotConfigured": "API 未配置",
    "common.all": "全部",
    "common.safe": "安全",
    "common.unsafe": "不安全",
    "status.uploaded": "已上传",
    "status.uploadError": "上传失败",
    "status.batchRunning": "批量评估中",
    "status.batchDone": "批量完成",
    "status.batchError": "批量失败",
    "batch.noGold": "未发现 gold 标签。报告已生成，但不包含聚合准确率指标。",
    "batch.ready": "已准备好批量评估。",
    "batch.datasetReady": "已选择生成评测集，可直接批量评估。",
    "batch.noDatasets": "未发现已生成的评测集。",
    "batch.noValidCases": "未找到有效案例",
    "batch.uploadHint": "请上传 JSON、JSONL，或包含 cases 数组的对象。",
    "batch.noUpload": "暂无上传批次",
    "batch.placeholder": "批量指标和报告下载链接会显示在这里。",
    "nav.home": "主页",
    "nav.evaluate": "Agent轨迹安全评估",
    "nav.model": "Guard Model调配",
    "nav.enchant": "安全能力附魔",
    "home.slogan": "让每一次 Agent 行动都留下可审计的安全证据",
    "home.meta": "AgentDoG taxonomy · CPU-first baseline · API validation · local model training hooks",
    "home.startEval": "开始评估",
    "home.openModel": "模型调配",
    "home.openEnchant": "安全附魔",
    "home.panel1.title": "轨迹级诊断",
    "home.panel1.body": "从用户目标、工具调用、外部 observation、最终回答中抽取风险证据。",
    "home.panel2.title": "低算力优先",
    "home.panel2.body": "默认离线规则与压缩启发式，可接入第三方 API，不强依赖 GPU。",
    "home.panel3.title": "赛前可调配",
    "home.panel3.body": "支持大规模数据生成、训练预检、模型运行方式切换和报告输出。",
    "home.panel4.title": "安全能力附魔",
    "home.panel4.body": "用当前 Guard Model 过滤、打分和奖励其他基座模型，生成更安全的策略模型。",
    "eval.title": "Agent轨迹安全评估",
    "eval.case": "案例",
    "eval.mode": "模式",
    "eval.judge": "判定器",
    "eval.run": "运行评估",
    "eval.apiNote": "第三方 API 推理只在服务端执行，API key 不会发送到浏览器。",
    "eval.input": "轨迹输入",
    "eval.report": "风险报告",
    "eval.reportHint": "运行评估后查看安全证据和成本。",
    "eval.apiInference": "第三方 API 推理",
    "eval.uploadTitle": "上传 JSON / JSONL",
    "eval.uploadHint": "点击选择文件，或拖拽轨迹文件到这里",
    "eval.datasetTitle": "选择已有生成评测集",
    "eval.datasetPlaceholder": "不使用已有评测集",
    "eval.datasetHint": "选择后可直接进行批量评估，无需重新上传文件。",
    "eval.refreshDatasets": "刷新列表",
    "eval.batchRun": "批量评估",
    "eval.endpoint": "端点",
    "eval.key": "密钥",
    "eval.pricing": "计费",
    "eval.label": "标签",
    "eval.decision": "决策",
    "eval.confidence": "置信度",
    "eval.taxonomy": "风险分类",
    "eval.riskSource": "风险来源",
    "eval.failureMode": "失败模式",
    "eval.harmType": "危害类型",
    "eval.cost": "成本",
    "eval.strategy": "策略",
    "eval.modelCalls": "模型调用",
    "eval.inputTokens": "输入 Token",
    "eval.latency": "延迟",
    "eval.compression": "压缩率",
    "eval.estimatedCost": "估算费用",
    "eval.notRun": "未运行",
    "eval.runtimeModel": "运行模型",
    "eval.runtimeEndpoint": "运行端点",
    "eval.lastMode": "上次模式",
    "eval.evidenceTimeline": "证据时间线",
    "eval.guardHook": "在线 Guard Hook",
    "eval.remoteModelNotConfigured": "远程模型未配置",
    "eval.keyPresent": "服务端密钥已配置",
    "eval.keyMissing": "缺少密钥",
    "eval.pricingMissing": "未配置计费",
    "eval.remoteCall": "次远程调用",
    "eval.ruleEarlyExit": "规则提前退出",
    "model.title": "Guard Model调配",
    "model.current": "当前使用模型",
    "model.mode": "使用方式",
    "model.localName": "本地模型名",
    "model.modelName": "使用模型",
    "model.apiMode": "模型 API",
    "model.localMode": "本地部署运行",
    "model.apiModePill": "API 模式",
    "model.localModePill": "本地模式",
    "model.apply": "应用配置",
    "model.apiModel": "API 模型",
    "model.localGpu": "本地 GPU",
    "model.trainer": "训练依赖",
    "model.dataGen": "数据生成设定",
    "model.dataGenHint": "支持 10K 级合成轨迹导出",
    "model.config": "配置文件",
    "model.scale": "生成规模",
    "model.genBackend": "生成后端",
    "model.backendDeterministic": "本地 Planner",
    "model.backendLlm": "LLM 轨迹生成",
    "model.semanticRepair": "语义修复",
    "model.repairLlm": "LLM self-repair",
    "model.repairNone": "不修复",
    "model.repairStatic": "本地兜底",
    "model.repairHybrid": "LLM 后本地兜底",
    "model.repairRounds": "修复轮数",
    "model.backend": "后端",
    "model.currentCase": "当前样本",
    "model.completedCases": "已完成",
    "model.rejectedCases": "失败/拒收",
    "model.scenario": "场景",
    "model.label": "标签",
    "model.evalSet": "评测集",
    "model.sftDataset": "SFT 数据集",
    "model.rlDataset": "RL / DPO / GRPO 数据集",
    "model.generate": "运行数据生成",
    "model.loadConfig": "加载配置",
    "model.generationStatus": "生成状态",
    "model.output": "输出目录",
    "model.evalArtifact": "评测集",
    "model.qualityReport": "质量报告",
    "model.rejectedSamples": "拒收样本",
    "model.trainingRejected": "训练过滤样本",
    "model.training": "本地微调训练",
    "model.localOnly": "仅本地部署模式可用",
    "model.rlAlgorithm": "RL 算法",
    "model.dataDir": "数据目录",
    "model.runSft": "运行 SFT",
    "model.runSftRl": "运行 SFT + RL",
    "model.autoSwitch": "完成后自动切换",
    "model.trainingReady": "本地部署训练预检",
    "model.trainingNeedsLocal": "切换到本地部署后可启用训练",
    "model.noFineTuned": "尚未注册微调模型",
    "model.switch": "切换",
    "model.trainerMissing": "缺少可选依赖",
    "enchant.title": "安全能力附魔",
    "enchant.guardTitle": "当前 Guard Model",
    "enchant.guardHint": "作为数据过滤器、语义 judge 和安全 reward",
    "enchant.roles": "训练角色",
    "enchant.roleList": "过滤 / 判定 / 奖励",
    "enchant.rewardFormula": "AgentDoG-style 奖励",
    "enchant.normalBenign": "普通良性",
    "enchant.attackedBenign": "被攻击良性",
    "enchant.malicious": "恶意任务",
    "enchant.configTitle": "目标模型训练设定",
    "enchant.configHint": "对其他 policy/base model 做安全蒸馏或 RL 对齐",
    "enchant.targetProfile": "目标模型 Profile",
    "enchant.algorithm": "训练方式",
    "enchant.maxSamples": "样本上限",
    "enchant.targetBase": "目标基座模型",
    "enchant.safetyWeight": "安全权重",
    "enchant.utilityWeight": "效用权重",
    "enchant.autoRegister": "完成后登记附魔模型",
    "enchant.run": "运行安全能力附魔",
    "enchant.jobTitle": "附魔任务状态",
    "enchant.plan": "训练计划",
    "enchant.modelsTitle": "已附魔目标模型",
    "enchant.modelsHint": "这些是被 Guard 迁移安全能力后的 policy/base model",
    "enchant.noModels": "尚未登记附魔模型",
  },
  en: {
    "common.checkingApi": "Checking API",
    "common.loading": "Loading",
    "common.idle": "Idle",
    "common.ready": "Ready",
    "common.running": "Running",
    "common.done": "Done",
    "common.error": "Error",
    "common.available": "available",
    "common.notAvailable": "not available",
    "common.notConfigured": "not configured",
    "common.model": "Model",
    "common.noData": "No data",
    "common.apiReady": "API Ready",
    "common.apiMissingKey": "API Missing Key",
    "common.apiNotConfigured": "API Not Configured",
    "common.all": "all",
    "common.safe": "safe",
    "common.unsafe": "unsafe",
    "status.uploaded": "Uploaded",
    "status.uploadError": "Upload Error",
    "status.batchRunning": "Batch Running",
    "status.batchDone": "Batch Done",
    "status.batchError": "Batch Error",
    "batch.noGold": "No gold labels found. Reports were generated without aggregate accuracy metrics.",
    "batch.ready": "Ready for batch evaluation.",
    "batch.datasetReady": "Generated eval dataset selected. Ready for batch evaluation.",
    "batch.noDatasets": "No generated eval datasets found.",
    "batch.noValidCases": "No valid cases found",
    "batch.uploadHint": "Upload JSON, JSONL, or an object with a cases array.",
    "batch.noUpload": "No uploaded batch",
    "batch.placeholder": "Batch metrics and report links will appear here.",
    "nav.home": "Home",
    "nav.evaluate": "Agent Trace Safety",
    "nav.model": "Guard Model Ops",
    "nav.enchant": "Safety Enchantment",
    "home.slogan": "Auditable safety evidence for every agent action",
    "home.meta": "AgentDoG taxonomy · CPU-first baseline · API validation · local model training hooks",
    "home.startEval": "Start Evaluation",
    "home.openModel": "Model Ops",
    "home.openEnchant": "Safety Enchant",
    "home.panel1.title": "Trajectory Diagnosis",
    "home.panel1.body": "Extract risk evidence from goals, tool calls, observations, and final answers.",
    "home.panel2.title": "Low Compute First",
    "home.panel2.body": "Offline rules and compression heuristics by default, with optional third-party API validation.",
    "home.panel3.title": "Contest Ready",
    "home.panel3.body": "Large-scale data export, training preflight, model switching, and experiment reports.",
    "home.panel4.title": "Safety Enchantment",
    "home.panel4.body": "Use the current Guard Model to filter, score, and reward other base models.",
    "eval.title": "Agent Trace Safety Evaluation",
    "eval.case": "Case",
    "eval.mode": "Mode",
    "eval.judge": "Judge",
    "eval.run": "Run Evaluation",
    "eval.apiNote": "Third-party API inference runs server-side. API keys never reach the browser.",
    "eval.input": "Trajectory Input",
    "eval.report": "Risk Report",
    "eval.reportHint": "Run an evaluation to inspect safety evidence and cost.",
    "eval.apiInference": "Third-party API Inference",
    "eval.uploadTitle": "Upload JSON / JSONL",
    "eval.uploadHint": "Click to choose files, or drop trajectory files here",
    "eval.datasetTitle": "Generated Eval Dataset",
    "eval.datasetPlaceholder": "Do not use generated dataset",
    "eval.datasetHint": "Select a generated eval set to run batch evaluation without uploading again.",
    "eval.refreshDatasets": "Refresh",
    "eval.batchRun": "Batch Evaluate",
    "eval.endpoint": "Endpoint",
    "eval.key": "Key",
    "eval.pricing": "Pricing",
    "eval.label": "Label",
    "eval.decision": "Decision",
    "eval.confidence": "Confidence",
    "eval.taxonomy": "Taxonomy",
    "eval.riskSource": "Risk Source",
    "eval.failureMode": "Failure Mode",
    "eval.harmType": "Harm Type",
    "eval.cost": "Cost",
    "eval.strategy": "Strategy",
    "eval.modelCalls": "Model Calls",
    "eval.inputTokens": "Input Tokens",
    "eval.latency": "Latency",
    "eval.compression": "Compression",
    "eval.estimatedCost": "Est. Cost",
    "eval.notRun": "Not run",
    "eval.runtimeModel": "Runtime Model",
    "eval.runtimeEndpoint": "Runtime Endpoint",
    "eval.lastMode": "Last Mode",
    "eval.evidenceTimeline": "Evidence Timeline",
    "eval.guardHook": "Online Guard Hook",
    "eval.remoteModelNotConfigured": "Remote model not configured",
    "eval.keyPresent": "server-side key present",
    "eval.keyMissing": "missing",
    "eval.pricingMissing": "not configured",
    "eval.remoteCall": "remote call",
    "eval.ruleEarlyExit": "rule early-exit",
    "model.title": "Guard Model Ops",
    "model.current": "Current Model",
    "model.mode": "Serving Mode",
    "model.localName": "Local Model",
    "model.modelName": "Model",
    "model.apiMode": "Model API",
    "model.localMode": "Local Deployment",
    "model.apiModePill": "API MODE",
    "model.localModePill": "LOCAL MODE",
    "model.apply": "Apply",
    "model.apiModel": "API Model",
    "model.localGpu": "Local GPU",
    "model.trainer": "Trainer",
    "model.dataGen": "Data Generation",
    "model.dataGenHint": "10K-ready synthetic trajectory export",
    "model.config": "Config",
    "model.scale": "Scale",
    "model.genBackend": "Generation Backend",
    "model.backendDeterministic": "Local Planner",
    "model.backendLlm": "LLM Trajectory",
    "model.semanticRepair": "Semantic Repair",
    "model.repairLlm": "LLM self-repair",
    "model.repairNone": "None",
    "model.repairStatic": "Local fallback",
    "model.repairHybrid": "LLM then local",
    "model.repairRounds": "Repair Rounds",
    "model.backend": "Backend",
    "model.currentCase": "Current Case",
    "model.completedCases": "Completed",
    "model.rejectedCases": "Rejected",
    "model.scenario": "Scenario",
    "model.label": "Label",
    "model.evalSet": "Eval set",
    "model.sftDataset": "SFT dataset",
    "model.rlDataset": "RL / DPO / GRPO dataset",
    "model.generate": "Generate Data",
    "model.loadConfig": "Load Config",
    "model.generationStatus": "Generation Status",
    "model.output": "Output",
    "model.evalArtifact": "Eval",
    "model.qualityReport": "Quality report",
    "model.rejectedSamples": "Rejected samples",
    "model.trainingRejected": "Training filtered",
    "model.training": "Local Fine-tuning",
    "model.localOnly": "Local deployment only",
    "model.rlAlgorithm": "RL Algorithm",
    "model.dataDir": "Data Dir",
    "model.runSft": "Run SFT",
    "model.runSftRl": "Run SFT + RL",
    "model.autoSwitch": "Auto switch after completion",
    "model.trainingReady": "Local deployment training preflight",
    "model.trainingNeedsLocal": "Switch to local deployment to enable training",
    "model.noFineTuned": "No fine-tuned model registered",
    "model.switch": "Switch",
    "model.trainerMissing": "missing optional deps",
    "enchant.title": "Safety Enchantment",
    "enchant.guardTitle": "Current Guard Model",
    "enchant.guardHint": "Used as data filter, semantic judge, and safety reward",
    "enchant.roles": "Training Roles",
    "enchant.roleList": "filter / judge / reward",
    "enchant.rewardFormula": "AgentDoG-style Reward",
    "enchant.normalBenign": "Normal benign",
    "enchant.attackedBenign": "Attacked benign",
    "enchant.malicious": "Malicious task",
    "enchant.configTitle": "Target Model Training",
    "enchant.configHint": "Safety distillation or RL alignment for another policy/base model",
    "enchant.targetProfile": "Target Profile",
    "enchant.algorithm": "Training Method",
    "enchant.maxSamples": "Max Samples",
    "enchant.targetBase": "Target Base Model",
    "enchant.safetyWeight": "Safety Weight",
    "enchant.utilityWeight": "Utility Weight",
    "enchant.autoRegister": "Register enchanted model",
    "enchant.run": "Run Safety Enchantment",
    "enchant.jobTitle": "Enchantment Job",
    "enchant.plan": "Training Plan",
    "enchant.modelsTitle": "Enchanted Target Models",
    "enchant.modelsHint": "Policy/base models improved by transferring guard safety.",
    "enchant.noModels": "No enchanted models registered",
  },
};

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function postJson(url, body) {
  return fetchJson(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
}

function setStatus(text) {
  statusEl.textContent = text;
}

function setText(selector, value) {
  const element = $(selector);
  if (element) {
    element.textContent = value ?? "-";
  }
}

function setVisible(selector, visible) {
  const element = $(selector);
  if (element) {
    element.classList.toggle("is-hidden", !visible);
  }
}

function t(key) {
  return copy[language][key] || copy.en[key] || key;
}

function formatLabel(value) {
  if (!value || value === "-") {
    return "-";
  }
  return String(value)
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatCost(value, note) {
  const number = Number(value || 0);
  if (!number) {
    return note && !String(note).startsWith("offline") ? "$0.000000 · pricing not set" : "$0.000000";
  }
  return `$${number.toFixed(6)}`;
}

function applyLanguage() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  langToggle.textContent = language === "zh" ? "EN" : "中";
  $$("[data-i18n]").forEach((element) => {
    const key = element.dataset.i18n;
    element.textContent = copy[language][key] || element.textContent;
  });
  refreshScenarioOptionLabels();
  if (runtimeInfo) {
    renderRuntime(runtimeInfo);
  }
  if (modelInfo) {
    renderModel(modelInfo);
  }
  if (enchantmentInfo) {
    renderEnchantmentStatus(enchantmentInfo);
  }
  renderEvalDatasetOptions(evalDatasets, evalDatasetSelect?.value || "");
}

function showPage(pageName) {
  const resolved = ["home", "evaluate", "model", "enchant"].includes(pageName) ? pageName : "home";
  $$(".page").forEach((page) => page.classList.remove("active"));
  $(`#${resolved}Page`).classList.add("active");
  $$(".nav-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.page === resolved));
  if (location.hash !== `#${resolved}`) {
    history.replaceState(null, "", `#${resolved}`);
  }
  window.scrollTo(0, 0);
  if (resolved === "model") {
    refreshModelStatus().catch(console.error);
  }
  if (resolved === "enchant") {
    refreshEnchantmentStatus().catch(console.error);
  }
}

function renderCase(rawCase) {
  currentCase = rawCase;
  caseInput.value = JSON.stringify(rawCase, null, 2);
  setText("#caseMeta", `${rawCase.metadata?.scenario || "scenario"} · ${rawCase.gold?.label || "unlabeled"}`);
}

function renderRuntime(info) {
  runtimeInfo = info;
  const api = info.api || {};
  const badge = $("#apiStatusBadge");
  badge.classList.remove("ready", "missing", "warn");
  if (api.configured && api.key_present) {
    badge.textContent = t("common.apiReady");
    badge.classList.add("ready");
  } else if (api.configured) {
    badge.textContent = t("common.apiMissingKey");
    badge.classList.add("warn");
  } else {
    badge.textContent = t("common.apiNotConfigured");
    badge.classList.add("missing");
  }
  setText("#apiModel", api.model ? `${t("common.model")}: ${api.model}` : t("eval.remoteModelNotConfigured"));
  setText("#apiEndpoint", api.api_base || "-");
  setText("#apiKeyState", api.key_present ? t("eval.keyPresent") : t("eval.keyMissing"));
  const pricing = api.pricing || {};
  setText(
    "#apiPricing",
    pricing.configured ? `$${pricing.input_per_1m_usd}/1M in · $${pricing.output_per_1m_usd}/1M out` : t("eval.pricingMissing"),
  );
  setText("#runtimeModel", api.model || "-");
  setText("#runtimeEndpoint", api.api_base || "-");
  if (info.model) {
    renderModel(info.model);
  }
  populateTargetProfiles(info.model_profiles || []);
}

function renderResult(result) {
  const report = result.report;
  const runtime = result.runtime || {};
  const api = runtime.api || runtimeInfo?.api || {};
  setText("#labelValue", report.label);
  setText("#decisionValue", report.decision);
  setText("#confidenceValue", report.confidence.toFixed(2));
  setText("#riskSource", formatLabel(report.risk_source));
  setText("#failureMode", formatLabel(report.failure_mode));
  setText("#harmType", formatLabel(report.harm_type));
  setText("#reasonValue", report.reason);
  setText("#judgeUsed", runtime.judge || judgeSelect.value);
  setText("#strategyValue", report.cost.strategy || runtime.strategy || "-");
  setText("#modelCalls", report.cost.model_calls);
  setText("#inputTokens", report.cost.input_tokens);
  setText("#latency", `${report.cost.latency_ms} ms`);
  setText("#compression", report.cost.compression_ratio);
  setText("#estimatedCost", formatCost(report.cost.estimated_cost_usd, report.cost.pricing_note));
  setText("#runtimeModel", api.model || "-");
  setText("#runtimeEndpoint", api.api_base || "-");
  setText("#runtimeMode", `${runtime.judge || judgeSelect.value} / ${runtime.mode || modeSelect.value}`);
  setText(
    "#apiCallBadge",
    report.cost.model_calls > 0 ? `${report.cost.model_calls} ${t("eval.remoteCall")}` : t("eval.ruleEarlyExit"),
  );

  const labelValue = $("#labelValue");
  labelValue.className = report.label === "unsafe" ? "label-unsafe" : "label-safe";

  const caseData = JSON.parse(caseInput.value);
  const evidence = new Set(report.evidence_steps || []);
  setText("#evidenceCount", language === "zh" ? `${evidence.size} 步` : `${evidence.size} ${evidence.size === 1 ? "step" : "steps"}`);
  const timeline = $("#evidenceList");
  timeline.innerHTML = "";
  for (const step of caseData.trajectory || []) {
    const item = document.createElement("div");
    item.className = evidence.has(step.step) ? "step evidence" : "step";
    const title = document.createElement("strong");
    title.textContent = `Step ${step.step} · ${step.role} · ${step.type}`;
    const body = document.createElement("div");
    body.className = "step-body";
    body.textContent = step.content || `${step.tool_name || ""} ${JSON.stringify(step.arguments || {})}`;
    item.append(title, body);
    timeline.append(item);
  }

  $("#guardOutput").textContent = JSON.stringify(result.guard, null, 2);
}

function renderBatchResult(result) {
  setText(
    "#uploadSummary",
    language === "zh"
      ? `${result.samples} 条样本 · ${result.gold_samples} 条有标签 · ${result.judge}/${result.mode} · ${result.source || ""}`
      : `${result.samples} samples · ${result.gold_samples} labeled · ${result.judge}/${result.mode} · ${result.source || ""}`,
  );
  const metrics = result.metrics || {};
  if (Object.keys(metrics).length) {
    setText(
      "#batchMetrics",
      `Accuracy=${formatMetric(metrics.accuracy)} · Precision=${formatMetric(metrics.precision)} · Recall=${formatMetric(
        metrics.recall,
      )} · F-score=${formatMetric(metrics.f_score)} · est_cost=${formatCost(metrics.total_estimated_cost_usd)}`,
    );
  } else {
    setText("#batchMetrics", t("batch.noGold"));
  }
  setDownload("#downloadJson", result.downloads?.json);
  setDownload("#downloadMd", result.downloads?.markdown);
  setDownload("#downloadChart", result.downloads?.chart);
}

function setDownload(selector, href) {
  const link = $(selector);
  if (!href) {
    link.href = "#";
    link.classList.add("disabled");
    link.setAttribute("aria-disabled", "true");
    return;
  }
  link.href = href;
  link.target = "_blank";
  link.classList.remove("disabled");
  link.setAttribute("aria-disabled", "false");
}

function formatMetric(value) {
  return Number(value || 0).toFixed(4);
}

async function loadCases() {
  try {
    renderRuntime(await fetchJson("/api/runtime"));
  } catch (error) {
    setText("#apiStatusBadge", "Runtime Error");
    console.error(error);
  }
  await loadEvalDatasets();
  const cases = await fetchJson("/api/cases");
  setText("#homeDatasetCount", `${cases.length} seed cases`);
  caseSelect.innerHTML = "";
  for (const item of cases) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.scenario}/${item.gold_label} · ${shortCaseName(item.id)}`;
    caseSelect.append(option);
  }
  if (cases.length) {
    renderCase(await fetchJson(`/api/cases/${cases[0].id}`));
  }
}

async function loadEvalDatasets() {
  if (!evalDatasetSelect) {
    return;
  }
  const selected = evalDatasetSelect.value;
  const response = await fetchJson("/api/eval-datasets");
  evalDatasets = response.datasets || [];
  renderEvalDatasetOptions(evalDatasets, selected);
}

function renderEvalDatasetOptions(datasets, selectedPath = "") {
  if (!evalDatasetSelect) {
    return;
  }
  evalDatasetSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = t("eval.datasetPlaceholder");
  evalDatasetSelect.append(placeholder);
  if (!(datasets || []).length) {
    const empty = document.createElement("option");
    empty.value = "";
    empty.disabled = true;
    empty.textContent = t("batch.noDatasets");
    evalDatasetSelect.append(empty);
  }
  for (const dataset of datasets || []) {
    const option = document.createElement("option");
    option.value = dataset.path;
    option.textContent = datasetOptionLabel(dataset);
    option.dataset.samples = dataset.samples || 0;
    option.dataset.goldSamples = dataset.gold_samples || 0;
    evalDatasetSelect.append(option);
  }
  if (selectedPath && (datasets || []).some((dataset) => dataset.path === selectedPath)) {
    evalDatasetSelect.value = selectedPath;
  }
}

function datasetOptionLabel(dataset) {
  const labels = dataset.labels || {};
  const labelText = Object.keys(labels).length
    ? Object.entries(labels)
        .map(([label, count]) => `${label}:${count}`)
        .join(" ")
    : language === "zh"
      ? "无标签"
      : "unlabeled";
  const updated = dataset.modified ? new Date(dataset.modified).toLocaleString(language === "zh" ? "zh-CN" : "en-US") : "";
  return `${dataset.name || dataset.path} · ${dataset.samples || 0} rows · ${labelText}${updated ? ` · ${updated}` : ""}`;
}

function selectedEvalDataset() {
  const path = evalDatasetSelect?.value || "";
  return evalDatasets.find((dataset) => dataset.path === path) || null;
}

function handleEvalDatasetSelection() {
  clearDownloads();
  const dataset = selectedEvalDataset();
  if (!dataset) {
    setText(
      "#uploadSummary",
      uploadedCases.length
        ? language === "zh"
          ? `已上传 ${uploadedCases.length} 条案例`
          : `${uploadedCases.length} uploaded case${uploadedCases.length === 1 ? "" : "s"}`
        : t("batch.noUpload"),
    );
    setText("#batchMetrics", uploadedCases.length ? t("batch.ready") : t("batch.placeholder"));
    return;
  }
  uploadedCases = [];
  setText(
    "#uploadSummary",
    language === "zh"
      ? `${dataset.name} · ${dataset.samples} 条样本 · ${dataset.gold_samples} 条有标签`
      : `${dataset.name} · ${dataset.samples} samples · ${dataset.gold_samples} labeled`,
  );
  setText("#batchMetrics", dataset.gold_samples ? t("batch.datasetReady") : t("batch.noGold"));
  setStatus(t("common.ready"));
}

function shortCaseName(id) {
  return String(id).replace(/^case_/, "").slice(0, 30);
}

function scenarioLabel(scenario) {
  return scenarioLabels[language][scenario] || scenario;
}

function generationStatusLabel(status) {
  const labels = {
    zh: {
      starting: "启动中",
      running: "准备样本",
      requesting: "请求 LLM",
      validating: "校验输出",
      retrying: "重试生成",
      repairing: "LLM 自修复",
      accepted: "已接收",
      rejected: "已拒收",
      completed: "完成",
    },
    en: {
      starting: "Starting",
      running: "Preparing",
      requesting: "Requesting LLM",
      validating: "Validating",
      retrying: "Retrying",
      repairing: "LLM Repair",
      accepted: "Accepted",
      rejected: "Rejected",
      completed: "Completed",
    },
  };
  return labels[language][status] || status || "-";
}

async function refreshModelStatus() {
  modelInfo = await fetchJson("/api/guard-model");
  renderModel(modelInfo);
}

function renderModel(info) {
  modelInfo = info;
  const api = info.api || {};
  const local = info.local || {};
  $("#deploymentModeSelect").value = info.deployment_mode || "local";
  setText("#currentModelName", info.current_model || "-");
  setText("#modelServingType", info.serving_type === "model_api" ? t("model.apiMode") : t("model.localMode"));
  setText("#modelApiName", api.model || "-");
  setText("#localGpuState", local.gpu_available ? t("common.available") : t("common.notAvailable"));
  setText("#trainerState", local.training_packages_available ? t("common.available") : t("model.trainerMissing"));
  const pill = $("#modelStatusPill");
  pill.classList.remove("ready", "warn", "missing");
  pill.textContent = info.deployment_mode === "api" ? t("model.apiModePill") : t("model.localModePill");
  pill.classList.add(info.deployment_mode === "api" && !api.configured ? "warn" : "ready");
  syncModelNameField(info);
  renderFineTunedModels(info.fine_tuned_models || []);
  populateScenarioOptions(info.scenarios || []);
  syncLocalControls();
  renderEnchantmentGuard(info);
}

async function refreshEnchantmentStatus() {
  enchantmentInfo = await fetchJson("/api/safety-enchantment");
  renderEnchantmentStatus(enchantmentInfo);
}

function renderEnchantmentStatus(info) {
  enchantmentInfo = info;
  const guard = info.guard || modelInfo || {};
  renderEnchantmentGuard(guard);
  populateTargetProfiles(info.model_profiles || runtimeInfo?.model_profiles || []);
  renderEnchantedModels(info.enchanted_models || []);
  const pill = $("#enchantStatusPill");
  if (pill) {
    pill.classList.remove("ready", "warn", "missing");
    pill.textContent = guard.current_model ? t("common.ready") : t("common.notConfigured");
    pill.classList.add(guard.current_model ? "ready" : "warn");
  }
  const formula = info.reward_formula || {};
  setText("#formulaNormal", formula.normal_benign || "U");
  setText("#formulaAttacked", formula.attacked_benign || "0.5 * U + 0.25 * S + 0.25 * U * S");
  setText("#formulaMalicious", formula.malicious || "S");
}

function renderEnchantmentGuard(info) {
  if (!$("#enchantGuardModel")) {
    return;
  }
  setText("#enchantGuardModel", info?.current_model || "-");
  setText("#enchantGuardMode", info?.serving_type === "model_api" ? t("model.apiMode") : t("model.localMode"));
}

function populateTargetProfiles(profiles) {
  const select = $("#targetProfileSelect");
  if (!select) {
    return;
  }
  const current = select.value;
  const localProfiles = (profiles || []).filter((profile) => profile.provider === "huggingface");
  if (!localProfiles.length) {
    return;
  }
  select.innerHTML = "";
  for (const profile of localProfiles) {
    const option = document.createElement("option");
    option.value = profile.name;
    option.textContent = `${profile.name} · ${profile.role || profile.family || ""}`.trim();
    option.dataset.modelId = profile.model_id || "";
    select.append(option);
  }
  if (current && localProfiles.some((profile) => profile.name === current)) {
    select.value = current;
  }
  syncTargetBasePlaceholder();
}

function syncTargetBasePlaceholder() {
  const select = $("#targetProfileSelect");
  const input = $("#targetBaseModel");
  if (!select || !input) {
    return;
  }
  const selected = select.selectedOptions[0];
  input.placeholder = selected?.dataset.modelId || "profile default";
}

function syncModelNameField(info = modelInfo) {
  const apiMode = $("#deploymentModeSelect").value === "api";
  const input = $("#localModelInput");
  setText("#modelNameFieldLabel", apiMode ? t("model.modelName") : t("model.localName"));
  if (apiMode) {
    input.value = info?.api?.model || info?.current_model || "";
    input.readOnly = true;
    input.setAttribute("aria-readonly", "true");
  } else {
    input.value = info?.local?.model || "tracehound-local-heuristic";
    input.readOnly = false;
    input.removeAttribute("aria-readonly");
  }
  input.classList.toggle("readonly-field", apiMode);
}

function populateScenarioOptions(scenarios) {
  const select = $("#genScenario");
  if (select.options.length > 1 || !scenarios.length) {
    refreshScenarioOptionLabels();
    return;
  }
  for (const scenario of scenarios) {
    const option = document.createElement("option");
    option.value = scenario;
    option.dataset.scenario = scenario;
    option.textContent = scenarioLabel(scenario);
    select.append(option);
  }
}

function refreshScenarioOptionLabels() {
  $$("#genScenario option[data-scenario]").forEach((option) => {
    option.textContent = scenarioLabel(option.dataset.scenario);
  });
}

function syncLocalControls() {
  const localMode = $("#deploymentModeSelect").value === "local";
  setVisible("#trainingPanel", localMode);
  setVisible("#sftDatasetOption", localMode);
  setVisible("#rlDatasetOption", localMode);
  $("#includeSft").disabled = !localMode;
  $("#includeRl").disabled = !localMode;
  $("#runSftBtn").disabled = !localMode;
  $("#runSftRlBtn").disabled = !localMode;
  setText("#trainingModeHint", localMode ? t("model.trainingReady") : t("model.trainingNeedsLocal"));
  syncModelNameField(modelInfo);
}

function renderFineTunedModels(models) {
  const target = $("#fineTunedModels");
  target.innerHTML = "";
  setVisible("#fineTunedModels", models.length > 0);
  if (!models.length) {
    return;
  }
  for (const model of models) {
    const item = document.createElement("article");
    const label = document.createElement("span");
    label.textContent = `${model.id} · ${model.algorithm}`;
    const button = document.createElement("button");
    button.className = "secondary-button";
    button.type = "button";
    button.textContent = t("model.switch");
    button.addEventListener("click", async () => {
      renderModel(await postJson("/api/guard-model/switch", {model_id: model.id}));
    });
    item.append(label, button);
    target.append(item);
  }
}

function renderEnchantedModels(models) {
  const panel = $("#enchantedModelsPanel");
  const target = $("#enchantedModels");
  if (!panel || !target) {
    return;
  }
  target.innerHTML = "";
  setVisible("#enchantedModelsPanel", models.length > 0);
  for (const model of models) {
    const item = document.createElement("article");
    const label = document.createElement("span");
    label.textContent = `${model.id} · ${model.target_profile || model.algorithm} · ${model.path || ""}`;
    item.append(label);
    target.append(item);
  }
}

async function startDataGeneration() {
  const mode = $("#deploymentModeSelect").value;
  const includeSft = mode === "local" && $("#includeSft").checked;
  const includeRl = mode === "local" && $("#includeRl").checked;
  const scenario = $("#genScenario").value;
  const label = $("#genLabel").value;
  const job = await postJson("/api/data-generation", {
    count: Number($("#genCount").value || 10000),
    generation_backend: $("#genBackend").value,
    semantic_repair_backend: $("#semanticRepairBackend").value,
    semantic_repair_rounds: Number($("#semanticRepairRounds").value || 1),
    scenarios: scenario === "all" ? [] : [scenario],
    labels: label === "all" ? [] : [label],
    config_path: $("#genConfigPath").value || undefined,
    include_eval: $("#includeEval").checked,
    include_sft: includeSft,
    include_rl: includeRl,
  });
  renderGenerationJob(job);
  pollJob(job.id, renderGenerationJob);
}

async function loadGenerationConfig() {
  const path = $("#genConfigPath").value || "configs/generation.yaml";
  const response = await fetchJson(`/api/generation-config?path=${encodeURIComponent(path)}`);
  const config = response.config || {};
  if (config.count) {
    $("#genCount").value = config.count;
  }
  if (config.generation_backend) {
    $("#genBackend").value = config.generation_backend;
  }
  if (config.semantic_repair_backend) {
    $("#semanticRepairBackend").value = config.semantic_repair_backend;
  }
  if (config.semantic_repair_rounds !== undefined && config.semantic_repair_rounds !== null) {
    $("#semanticRepairRounds").value = config.semantic_repair_rounds;
  }
  if (Array.isArray(config.scenarios) && config.scenarios.length === 1) {
    $("#genScenario").value = config.scenarios[0];
  } else {
    $("#genScenario").value = "all";
  }
  if (Array.isArray(config.labels) && config.labels.length === 1) {
    $("#genLabel").value = config.labels[0];
  } else {
    $("#genLabel").value = "all";
  }
  $("#includeEval").checked = config.include_eval !== false;
  $("#includeSft").checked = Boolean(config.include_sft);
  $("#includeRl").checked = Boolean(config.include_rl);
  setVisible("#generationJobPanel", true);
  setVisible("#jobProgressTrack", false);
  setVisible("#artifactList", false);
  setText("#jobStep", language === "zh" ? "配置已加载" : "Config loaded");
  renderMessages("#jobMessages", [`loaded ${response.path}`, JSON.stringify(config)]);
  syncLocalControls();
}

async function loadUploadedFiles(files) {
  const parsed = [];
  for (const file of files) {
    const text = await file.text();
    parsed.push(...parseTrajectoryFile(text, file.name));
  }
  uploadedCases = parsed;
  if (evalDatasetSelect) {
    evalDatasetSelect.value = "";
  }
  clearDownloads();
  if (uploadedCases.length) {
    renderCase(uploadedCases[0]);
    setText(
      "#uploadSummary",
      language === "zh" ? `已上传 ${uploadedCases.length} 条案例` : `${uploadedCases.length} uploaded case${uploadedCases.length === 1 ? "" : "s"}`,
    );
    setText("#batchMetrics", t("batch.ready"));
    setStatus(t("status.uploaded"));
  } else {
    setText("#uploadSummary", t("batch.noValidCases"));
    setText("#batchMetrics", t("batch.uploadHint"));
  }
}

function parseTrajectoryFile(text, name) {
  const trimmed = text.trim();
  if (!trimmed) {
    return [];
  }
  if (name.endsWith(".jsonl") || (!trimmed.startsWith("{") && !trimmed.startsWith("["))) {
    return trimmed
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line))
      .flatMap(normalizeUploadedPayload);
  }
  return normalizeUploadedPayload(JSON.parse(trimmed));
}

function normalizeUploadedPayload(payload) {
  if (Array.isArray(payload)) {
    return payload.flatMap(normalizeUploadedPayload);
  }
  if (payload && Array.isArray(payload.cases)) {
    return payload.cases.flatMap(normalizeUploadedPayload);
  }
  if (payload && payload.case && payload.case.trajectory) {
    return [{...payload.case, gold: payload.gold || payload.case.gold || payload.label}];
  }
  if (payload && payload.trajectory) {
    return [payload];
  }
  return [];
}

function clearDownloads() {
  setDownload("#downloadJson", "");
  setDownload("#downloadMd", "");
  setDownload("#downloadChart", "");
}

async function runBatchEvaluation() {
  const datasetPath = evalDatasetSelect?.value || "";
  const cases = datasetPath ? [] : uploadedCases.length ? uploadedCases : [JSON.parse(caseInput.value)];
  batchRunBtn.disabled = true;
  setStatus(t("status.batchRunning"));
  try {
    const result = await postJson("/api/batch-evaluate", {
      mode: modeSelect.value,
      judge: judgeSelect.value,
      dataset_path: datasetPath || undefined,
      cases: datasetPath ? undefined : cases,
    });
    renderBatchResult(result);
    setStatus(t("status.batchDone"));
  } catch (error) {
    setStatus(t("status.batchError"));
    setText("#batchMetrics", error.message);
    console.error(error);
  } finally {
    batchRunBtn.disabled = false;
  }
}

async function startTraining(kind) {
  const job = await postJson("/api/training", {
    kind,
    algorithm: $("#rlAlgorithm").value,
    data_dir: $("#trainDataDir").value,
    auto_switch: $("#autoSwitchModel").checked,
  });
  renderTrainingJob(job);
  pollJob(job.id, renderTrainingJob, async () => {
    await refreshModelStatus();
  });
}

async function startSafetyEnchantment() {
  const maxSamples = Number($("#enchantMaxSamples").value || 0);
  const job = await postJson("/api/safety-enchantment", {
    algorithm: $("#enchantAlgorithm").value,
    target_model_profile: $("#targetProfileSelect").value,
    target_base_model: $("#targetBaseModel").value || undefined,
    data_dir: $("#enchantDataDir").value,
    output_dir: $("#enchantOutputDir").value,
    max_samples: maxSamples > 0 ? maxSamples : undefined,
    safety_weight: Number($("#safetyWeight").value || 0.5),
    utility_weight: Number($("#utilityWeight").value || 0.5),
    auto_register: $("#autoRegisterEnchant").checked,
  });
  renderEnchantmentJob(job);
  pollJob(job.id, renderEnchantmentJob, async () => {
    await refreshEnchantmentStatus();
  });
}

async function pollJob(jobId, render, onDone) {
  let done = false;
  while (!done) {
    await new Promise((resolve) => setTimeout(resolve, 650));
    const job = await fetchJson(`/api/jobs/${jobId}`);
    render(job);
    done = ["completed", "failed", "blocked", "requires_gpu"].includes(job.status);
  }
  if (onDone) {
    await onDone();
  }
}

function renderGenerationJob(job) {
  setVisible("#generationJobPanel", true);
  setText("#jobStep", `${job.status} · ${job.step}`);
  const progress = Number(job.progress || 0);
  setVisible("#jobProgressTrack", progress > 0);
  $("#jobProgressBar").style.width = `${progress}%`;
  renderGenerationLiveStats(job.synthesis || null);
  renderMessages("#jobMessages", job.messages || []);
  const artifacts = job.artifacts || {};
  const hasArtifacts = Boolean(
    artifacts.output_dir ||
      artifacts.eval ||
      artifacts.quality_report ||
      artifacts.rejected ||
      artifacts.training_rejected ||
      artifacts.sft ||
      artifacts.rl,
  );
  setVisible("#artifactList", hasArtifacts);
  setText("#artifactOutput", artifacts.output_dir || "-");
  setText("#artifactEval", artifacts.eval || "-");
  setText("#artifactQuality", artifacts.quality_report || "-");
  setText("#artifactRejected", artifacts.rejected || "-");
  setText("#artifactTrainingRejected", artifacts.training_rejected || "-");
  setText("#artifactSft", artifacts.sft || "-");
  setText("#artifactRl", artifacts.rl || "-");
  if (artifacts.output_dir) {
    $("#trainDataDir").value = artifacts.output_dir;
  }
}

function renderGenerationLiveStats(synthesis) {
  const visible = Boolean(synthesis && synthesis.backend);
  setVisible("#generationLiveStats", visible);
  if (!visible) {
    return;
  }
  const attempt = synthesis.attempt ? (language === "zh" ? ` · 第 ${synthesis.attempt} 次` : ` · attempt ${synthesis.attempt}`) : "";
  setText("#genLiveBackend", `${synthesis.backend || "-"} · ${generationStatusLabel(synthesis.status)}${attempt}`);
  const total = synthesis.total || 0;
  const current = synthesis.current || 0;
  const caseId = synthesis.case_id ? ` · ${synthesis.case_id}` : "";
  setText("#genLiveCurrent", total ? `${current}/${total}${caseId}` : "-");
  setText("#genLiveCompleted", synthesis.completed ?? 0);
  setText("#genLiveRejected", synthesis.rejected ?? 0);
}

function renderTrainingJob(job) {
  const progress = Number(job.progress || 0);
  setVisible("#trainProgressTrack", progress > 0);
  $("#trainProgressBar").style.width = `${progress}%`;
  renderMessages("#trainingMessages", job.messages || []);
}

function renderEnchantmentJob(job) {
  setVisible("#enchantJobPanel", true);
  setText("#enchantJobStep", `${job.status} · ${job.step}`);
  const progress = Number(job.progress || 0);
  setVisible("#enchantProgressTrack", progress > 0);
  $("#enchantProgressBar").style.width = `${progress}%`;
  renderMessages("#enchantMessages", job.messages || []);
  const artifacts = job.artifacts || {};
  const hasArtifacts = Boolean(artifacts.output_dir || artifacts.plan || (artifacts.commands || []).length);
  setVisible("#enchantArtifactList", hasArtifacts);
  setText("#enchantArtifactOutput", artifacts.output_dir || "-");
  setText("#enchantArtifactPlan", artifacts.plan || "-");
  renderCommandList(artifacts.commands || []);
}

function renderCommandList(commands) {
  const target = $("#enchantCommandList");
  if (!target) {
    return;
  }
  target.innerHTML = "";
  target.classList.toggle("is-hidden", !commands.length);
  for (const command of commands) {
    const line = document.createElement("code");
    line.textContent = command;
    target.append(line);
  }
}

function renderMessages(selector, messages) {
  const target = $(selector);
  target.innerHTML = "";
  const visibleMessages = (messages || []).filter((message) => String(message || "").trim());
  target.classList.toggle("is-hidden", visibleMessages.length === 0);
  for (const message of visibleMessages.slice(-8)) {
    const line = document.createElement("div");
    line.textContent = String(message);
    target.append(line);
  }
}

caseSelect.addEventListener("change", async () => {
  setStatus(t("common.loading"));
  try {
    renderCase(await fetchJson(`/api/cases/${caseSelect.value}`));
    setStatus(t("common.ready"));
  } catch (error) {
    setStatus(t("common.error"));
    console.error(error);
  }
});

runBtn.addEventListener("click", async () => {
  setStatus(t("common.running"));
  runBtn.disabled = true;
  try {
    const result = await postJson("/api/evaluate", {
      mode: modeSelect.value,
      judge: judgeSelect.value,
      case: JSON.parse(caseInput.value),
    });
    renderResult(result);
    setStatus(t("common.done"));
  } catch (error) {
    setStatus(t("common.error"));
    setText("#reasonValue", error.message);
    $("#guardOutput").textContent = error.stack || String(error);
    console.error(error);
  } finally {
    runBtn.disabled = false;
  }
});

$$(".nav-tab").forEach((tab) => {
  tab.addEventListener("click", () => showPage(tab.dataset.page));
});

$$("[data-go]").forEach((button) => {
  button.addEventListener("click", () => showPage(button.dataset.go));
});

window.addEventListener("hashchange", () => showPage(location.hash.replace("#", "")));

langToggle.addEventListener("click", () => {
  language = language === "zh" ? "en" : "zh";
  applyLanguage();
});

$("#deploymentModeSelect").addEventListener("change", syncLocalControls);

$("#applyModelBtn").addEventListener("click", async () => {
  const deploymentMode = $("#deploymentModeSelect").value;
  renderModel(
    await postJson("/api/guard-model", {
      deployment_mode: deploymentMode,
      local_model: deploymentMode === "local" ? $("#localModelInput").value : undefined,
    }),
  );
});

$("#generateDataBtn").addEventListener("click", async () => {
  $("#generateDataBtn").disabled = true;
  setVisible("#generationJobPanel", true);
  try {
    await startDataGeneration();
  } catch (error) {
    renderMessages("#jobMessages", [error.message]);
  } finally {
    $("#generateDataBtn").disabled = false;
  }
});

$("#loadGenConfigBtn").addEventListener("click", () => loadGenerationConfig().catch((error) => renderMessages("#jobMessages", [error.message])));

$("#runSftBtn").addEventListener("click", () => startTraining("sft").catch((error) => renderMessages("#trainingMessages", [error.message])));
$("#runSftRlBtn").addEventListener("click", () =>
  startTraining("sft_rl").catch((error) => renderMessages("#trainingMessages", [error.message])),
);
$("#targetProfileSelect").addEventListener("change", syncTargetBasePlaceholder);
$("#runEnchantBtn").addEventListener("click", () => {
  $("#runEnchantBtn").disabled = true;
  setVisible("#enchantJobPanel", true);
  startSafetyEnchantment()
    .catch((error) => renderMessages("#enchantMessages", [error.message]))
    .finally(() => {
      $("#runEnchantBtn").disabled = false;
    });
});

dropZone.addEventListener("click", () => jsonUpload.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    jsonUpload.click();
  }
});
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  loadUploadedFiles([...event.dataTransfer.files]).catch((error) => {
    setStatus(t("status.uploadError"));
    setText("#batchMetrics", error.message);
  });
});
jsonUpload.addEventListener("change", () => {
  loadUploadedFiles([...jsonUpload.files]).catch((error) => {
    setStatus(t("status.uploadError"));
    setText("#batchMetrics", error.message);
  });
});
evalDatasetSelect.addEventListener("change", handleEvalDatasetSelection);
refreshDatasetsBtn.addEventListener("click", () => {
  refreshDatasetsBtn.disabled = true;
  loadEvalDatasets()
    .then(handleEvalDatasetSelection)
    .catch((error) => {
      setStatus(t("common.error"));
      setText("#batchMetrics", error.message);
    })
    .finally(() => {
      refreshDatasetsBtn.disabled = false;
    });
});
batchRunBtn.addEventListener("click", runBatchEvaluation);

applyLanguage();
showPage(location.hash.replace("#", "") || "home");
loadCases()
  .then(refreshModelStatus)
  .then(refreshEnchantmentStatus)
  .catch((error) => {
    setStatus(t("common.error"));
    console.error(error);
  });
