// ---- Chart.js light / NeoStats theme --------------------------------------
Chart.defaults.color = "#66708a";
Chart.defaults.font.family = "Plex Sans, system-ui, Segoe UI, sans-serif";
Chart.defaults.borderColor = "rgba(26,34,56,0.08)";

const NAVY = "#1a2238", GREEN = "#4caf50", STEEL = "#7e8aa6", AMBER = "#d9911f", RED = "#c0392b";

const api = (path, opts) => fetch(path, opts).then((r) => r.json());
let lastApplicant = null;
const charts = {};

// Inline line icons (no emoji) -----------------------------------------------
const ICON = {
  users: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="9" cy="8" r="3.2"/><path d="M3 20a6 6 0 0 1 12 0M16 5.5a3 3 0 0 1 0 5.8M21 20a5.5 5.5 0 0 0-4-5"/></svg>',
  grid: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>',
  alert: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>',
  cross: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/></svg>',
  gap: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="5" width="18" height="14" rx="2" stroke-dasharray="3 3"/></svg>',
};
const ALERT_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>';

// ---- Navigation ------------------------------------------------------------
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
tabs.forEach((t) => t.addEventListener("click", () => {
  tabs.forEach((x) => x.classList.remove("active"));
  panels.forEach((p) => p.classList.remove("active"));
  t.classList.add("active");
  document.getElementById(t.dataset.tab).classList.add("active");
  document.getElementById("page-title").textContent = t.dataset.title;
  document.getElementById("page-sub").textContent = t.dataset.sub;
}));

// ---- Chart helper ----------------------------------------------------------
function bar(canvasId, labels, data, color, horizontal = false) {
  if (charts[canvasId]) charts[canvasId].destroy();
  charts[canvasId] = new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: color, borderRadius: 2, maxBarThickness: 40 }] },
    options: {
      indexAxis: horizontal ? "y" : "x",
      responsive: true,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: NAVY } },
      scales: {
        x: { grid: { color: "rgba(26,34,56,0.06)" }, ticks: { font: { size: 11 } } },
        y: { grid: { color: "rgba(26,34,56,0.06)" }, ticks: { font: { size: 11 } } },
      },
    },
  });
}

// ---- EDA -------------------------------------------------------------------
async function loadEDA() {
  let d;
  try { d = await api("/api/eda"); } catch { return; }
  if (d.detail) { document.getElementById("eda-cards").innerHTML = `<p>${d.detail}</p>`; return; }
  const s = d.summary;
  const kpis = [
    [ICON.users, s.n_rows.toLocaleString(), "Applicants"],
    [ICON.grid, s.n_cols, "Columns"],
    [ICON.alert, s.default_rate_pct + "%", "Default rate"],
    [ICON.cross, s.target_counts.defaulted.toLocaleString(), "Defaulted"],
    [ICON.gap, d.data_quality.columns_with_missing, "Cols w/ missing"],
  ];
  document.getElementById("eda-cards").innerHTML = kpis.map(([ic, v, l]) =>
    `<div class="kpi"><div class="ico">${ic}</div><div class="value">${v}</div><div class="label">${l}</div></div>`).join("");
  document.getElementById("eda-findings").innerHTML =
    "<h3>Key findings</h3><ul>" + d.key_findings.map((f) => `<li>${f}</li>`).join("") + "</ul>";

  const edu = d.insights.default_rate_by_education;
  bar("chart-education", edu.map((x) => x.category), edu.map((x) => x.default_rate_pct), GREEN, true);
  const age = d.insights.default_rate_by_age_band;
  bar("chart-age", age.map((x) => x.category), age.map((x) => x.default_rate_pct), NAVY);
  const inc = d.insights.default_rate_by_income_type;
  bar("chart-income-type", inc.map((x) => x.category), inc.map((x) => x.default_rate_pct), STEEL, true);
  const con = d.insights.default_rate_by_contract;
  bar("chart-contract", con.map((x) => x.category), con.map((x) => x.default_rate_pct), GREEN);
}

// ---- Model badge -----------------------------------------------------------
async function loadBadge() {
  try {
    const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    if (r.model_metrics) document.getElementById("badge-auc").textContent = "ROC-AUC " + r.model_metrics.roc_auc.toFixed(3);
  } catch {}
}

// ---- Ring gauge ------------------------------------------------------------
const R = 72, CIRC = 2 * Math.PI * R;
function setGauge(prob, band) {
  const colors = { Low: "#2e9b50", Medium: "#d9911f", High: "#c0392b" };
  const ring = document.getElementById("ring-fill");
  ring.style.stroke = colors[band] || NAVY;
  ring.style.strokeDasharray = CIRC;
  const shown = Math.min(prob / 0.6, 1); // scale 0–60% so small probs are visible
  ring.style.strokeDashoffset = CIRC * (1 - shown);
  document.getElementById("prob-value").textContent = (prob * 100).toFixed(1) + "%";
  const bt = document.getElementById("band-value");
  bt.textContent = band + " risk"; bt.className = "band-tag band-" + band;
}

// ---- Predict ---------------------------------------------------------------
function gatherApplicant() {
  const num = (id) => { const v = document.getElementById(id).value; return v === "" ? null : Number(v); };
  const str = (id) => document.getElementById(id).value || null;
  const age = num("AGE_YEARS"), emp = num("YEARS_EMP");
  return {
    AMT_INCOME_TOTAL: num("AMT_INCOME_TOTAL"), AMT_CREDIT: num("AMT_CREDIT"),
    AMT_ANNUITY: num("AMT_ANNUITY"), AMT_GOODS_PRICE: num("AMT_GOODS_PRICE"),
    DAYS_BIRTH: age != null ? Math.round(-age * 365.25) : null,
    DAYS_EMPLOYED: emp != null ? Math.round(-emp * 365.25) : null,
    CNT_CHILDREN: num("CNT_CHILDREN"),
    EXT_SOURCE_1: num("EXT_SOURCE_1"), EXT_SOURCE_2: num("EXT_SOURCE_2"), EXT_SOURCE_3: num("EXT_SOURCE_3"),
    CODE_GENDER: str("CODE_GENDER"), NAME_CONTRACT_TYPE: str("NAME_CONTRACT_TYPE"),
    NAME_EDUCATION_TYPE: str("NAME_EDUCATION_TYPE"), FLAG_OWN_CAR: str("FLAG_OWN_CAR"),
  };
}

document.getElementById("btn-predict").addEventListener("click", async () => {
  lastApplicant = gatherApplicant();
  const clean = Object.fromEntries(Object.entries(lastApplicant).filter(([, v]) => v != null));
  const res = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(clean) });
  const box = document.getElementById("predict-result");
  box.classList.add("show");
  if (res.detail) { box.innerHTML = `<div class="error-card">${ALERT_ICON}<div>${res.detail}</div></div>`; return; }
  setGauge(res.default_probability, res.risk_band);
  const ks = res.model_metrics.ks_statistic;
  const recClass = { Approve: "Low", "Manual Review": "Medium", Decline: "High" }[res.recommendation] || "Medium";
  document.getElementById("model-metrics").innerHTML = `
    <div class="mc rec band-${recClass}"><div class="m-val">${res.recommendation}</div><div class="m-lab">Underwriting recommendation</div></div>
    <div class="mc"><div class="m-val">${res.model_metrics.roc_auc.toFixed(3)}</div><div class="m-lab">Model ROC-AUC</div></div>
    <div class="mc"><div class="m-val">${res.model_metrics.pr_auc.toFixed(3)}</div><div class="m-lab">Model PR-AUC</div></div>
    ${ks != null ? `<div class="mc"><div class="m-val">${ks.toFixed(3)}</div><div class="m-lab">KS statistic</div></div>` : ""}`;
});

// ---- Explain ---------------------------------------------------------------
document.getElementById("btn-explain").addEventListener("click", async () => {
  const list = document.getElementById("explain-list");
  if (!lastApplicant) { list.innerHTML = `<div class="error-card">${ALERT_ICON}<div>Run a prediction first (Risk Prediction tab).</div></div>`; return; }
  const clean = Object.fromEntries(Object.entries(lastApplicant).filter(([, v]) => v != null));
  list.innerHTML = `<div class="loading"><span class="spinner"></span> Computing SHAP values…</div>`;
  const res = await api("/api/explain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(clean) });
  if (res.detail) { list.innerHTML = `<div class="error-card">${ALERT_ICON}<div>${res.detail}</div></div>`; return; }
  const top = [...res.top_risk_increasing.slice(0, 6), ...res.top_risk_decreasing.slice(0, 6)].sort((a, b) => a.shap_value - b.shap_value);
  bar("chart-shap", top.map((c) => c.feature), top.map((c) => c.shap_value),
    top.map((c) => c.shap_value > 0 ? RED : "#2e9b50"), true);
  const row = (c, dir) => `<div class="driver ${dir}"><span>${c.explanation}</span><span class="sv">${c.shap_value > 0 ? "+" : ""}${c.shap_value.toFixed(3)}</span></div>`;
  list.innerHTML = `<div class="drivers">
    <div class="col"><h4>Increasing risk</h4>${res.top_risk_increasing.slice(0, 6).map((c) => row(c, "up")).join("")}</div>
    <div class="col"><h4>Decreasing risk</h4>${res.top_risk_decreasing.slice(0, 6).map((c) => row(c, "down")).join("")}</div>
  </div>`;
});

// ---- Rules -----------------------------------------------------------------
async function loadRules() {
  const res = await api("/api/rules");
  const tbody = document.querySelector("#rules-table tbody");
  if (res.detail) { tbody.innerHTML = `<tr><td colspan="5">${res.detail}</td></tr>`; return; }
  document.getElementById("rules-base").textContent =
    `Population base default rate: ${(res.base_default_rate * 100).toFixed(1)}%. Rules sorted by deviation from base.`;
  tbody.innerHTML = res.rules.map((r) => `<tr>
    <td><span class="pill ${r.risk_band}">${r.risk_band}</span></td>
    <td class="mono">${r.rule}</td>
    <td class="num">${(r.default_rate * 100).toFixed(1)}%</td>
    <td class="num">${r.support.toLocaleString()}</td>
    <td class="lift">${r.lift ?? "-"}×</td></tr>`).join("");
}

// ---- Chat ------------------------------------------------------------------
async function loadChatExamples() {
  const { examples } = await api("/api/chat/examples");
  const box = document.getElementById("chat-examples");
  box.innerHTML = examples.map((q) => `<button class="chip">${q}</button>`).join("");
  box.querySelectorAll(".chip").forEach((b) =>
    b.addEventListener("click", () => { document.getElementById("chat-q").value = b.textContent; askChat(); }));
}

async function askChat() {
  const q = document.getElementById("chat-q").value.trim();
  if (!q) return;
  const out = document.getElementById("chat-answer");
  out.innerHTML = `<div class="loading"><span class="spinner"></span> Querying the data…</div>`;
  const res = await api("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
  if (res.error) { out.innerHTML = `<div class="error-card">${ALERT_ICON}<div><b>Error</b><br>${res.error}</div></div>`; return; }
  let table = "";
  if (res.rows && res.rows.length) {
    const cols = res.columns;
    table = `<div class="table-card"><table><thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
      res.rows.slice(0, 20).map((r) => `<tr>${cols.map((c) => `<td class="num">${r[c]}</td>`).join("")}</tr>`).join("") +
      `</tbody></table></div>`;
  }
  const src = res.source === "gemini" ? "Generated by Gemini" : "Offline query generator";
  out.innerHTML = `<div class="answer-card">
    <div class="ans">${res.answer || ""}</div>
    <div class="src">${src} · prompt ${res.prompt_version || ""}</div>
    <div class="sql-block">${res.sql || ""}</div>
    ${table}</div>`;
}
document.getElementById("btn-chat").addEventListener("click", askChat);
document.getElementById("chat-q").addEventListener("keydown", (e) => { if (e.key === "Enter") askChat(); });

// ---- Init ------------------------------------------------------------------
loadEDA(); loadRules(); loadChatExamples(); loadBadge();
