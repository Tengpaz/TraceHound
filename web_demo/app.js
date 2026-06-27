const caseSelect = document.querySelector("#caseSelect");
const modeSelect = document.querySelector("#modeSelect");
const judgeSelect = document.querySelector("#judgeSelect");
const caseInput = document.querySelector("#caseInput");
const runBtn = document.querySelector("#runBtn");
const statusEl = document.querySelector("#status");

let currentCase = null;
let runtimeInfo = null;

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function setStatus(text) {
  statusEl.textContent = text;
}

function setText(id, value) {
  document.querySelector(id).textContent = value ?? "-";
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

function renderCase(rawCase) {
  currentCase = rawCase;
  caseInput.value = JSON.stringify(rawCase, null, 2);
}

function renderRuntime(info) {
  runtimeInfo = info;
  const api = info.api || {};
  const badge = document.querySelector("#apiStatusBadge");
  badge.classList.remove("ready", "missing");
  if (api.configured && api.key_present) {
    badge.textContent = "API Ready";
    badge.classList.add("ready");
  } else if (api.configured) {
    badge.textContent = "API Missing Key";
    badge.classList.add("missing");
  } else {
    badge.textContent = "API Not Configured";
    badge.classList.add("missing");
  }
  setText("#apiModel", api.model ? `Model: ${api.model}` : "Remote model not configured");
  setText("#apiEndpoint", api.api_base || "-");
  setText("#apiKeyState", api.key_present ? "server-side key present" : "missing");
  setText("#runtimeModel", api.model || "-");
  setText("#runtimeEndpoint", api.api_base || "-");
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
  setText("#runtimeModel", api.model || "-");
  setText("#runtimeEndpoint", api.api_base || "-");
  setText("#runtimeMode", `${runtime.judge || judgeSelect.value} / ${runtime.mode || modeSelect.value}`);
  setText("#apiCallBadge", report.cost.model_calls > 0 ? `${report.cost.model_calls} remote call` : "rule early-exit");

  const labelValue = document.querySelector("#labelValue");
  labelValue.className = report.label === "unsafe" ? "label-unsafe" : "label-safe";

  const caseData = JSON.parse(caseInput.value);
  const evidence = new Set(report.evidence_steps || []);
  setText("#evidenceCount", `${evidence.size} ${evidence.size === 1 ? "step" : "steps"}`);
  const timeline = document.querySelector("#evidenceList");
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

  document.querySelector("#guardOutput").textContent = JSON.stringify(result.guard, null, 2);
}

async function loadCases() {
  try {
    renderRuntime(await fetchJson("/api/runtime"));
  } catch (error) {
    setText("#apiStatusBadge", "Runtime Error");
    console.error(error);
  }
  const cases = await fetchJson("/api/cases");
  caseSelect.innerHTML = "";
  for (const item of cases) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.id} · ${item.gold_label}`;
    caseSelect.append(option);
  }
  if (cases.length) {
    const first = await fetchJson(`/api/cases/${cases[0].id}`);
    renderCase(first);
  }
}

caseSelect.addEventListener("change", async () => {
  setStatus("Loading");
  try {
    renderCase(await fetchJson(`/api/cases/${caseSelect.value}`));
    setStatus("Ready");
  } catch (error) {
    setStatus("Error");
    console.error(error);
  }
});

runBtn.addEventListener("click", async () => {
  setStatus("Running");
  runBtn.disabled = true;
  try {
    const body = {
      mode: modeSelect.value,
      judge: judgeSelect.value,
      case: JSON.parse(caseInput.value),
    };
    const result = await fetchJson("/api/evaluate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    renderResult(result);
    setStatus("Done");
  } catch (error) {
    setStatus("Error");
    setText("#reasonValue", error.message);
    document.querySelector("#guardOutput").textContent = error.stack || String(error);
    console.error(error);
  } finally {
    runBtn.disabled = false;
  }
});

loadCases().catch((error) => {
  setStatus("Error");
  console.error(error);
});
