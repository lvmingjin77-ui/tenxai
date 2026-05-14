const form = document.getElementById("backtest-form");
const datasetSummary = document.getElementById("dataset-summary");
const datasetSelect = document.getElementById("dataset-select");
const notes = document.getElementById("notes");
const sourceList = document.getElementById("source-list");
const datasetLibrary = document.getElementById("dataset-library");
const datasetDetail = document.getElementById("dataset-detail");
const metricGrid = document.getElementById("metric-grid");
const resultRange = document.getElementById("result-range");
const decisionList = document.getElementById("decision-list");
const tradeBody = document.getElementById("trade-body");
const chart = document.getElementById("equity-chart");
const runButton = document.getElementById("run-button");
let datasetCatalog = [];

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function money(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(Number(value || 0));
}

function titleCase(value) {
  return String(value || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ");
}

function renderWeightChips(targetWeights, cashWeight) {
  const chips = Object.entries(targetWeights || {})
    .map(
      ([symbol, value]) => `
        <span class="weight-chip">
          <strong>${symbol}</strong>
          <span>${pct(value)}</span>
        </span>
      `,
    )
    .join("");
  const cashChip = `
    <span class="weight-chip weight-chip-cash">
      <strong>Cash</strong>
      <span>${pct(cashWeight || 0)}</span>
    </span>
  `;
  return chips || cashWeight ? `${chips}${cashChip}` : `<span class="weight-chip weight-chip-cash"><strong>Cash</strong><span>100.00%</span></span>`;
}

function setDefaults(config) {
  for (const [key, value] of Object.entries(config)) {
    const input = form.elements.namedItem(key);
    if (input) {
      input.value = value;
    }
  }
}

function renderDatasetOptions(datasets, defaultDatasetId) {
  datasetCatalog = datasets;
  datasetSelect.disabled = datasets.length <= 1;
  datasetSelect.innerHTML = datasets
    .map(
      (item) => `
        <option value="${item.dataset_id}" ${item.dataset_id === defaultDatasetId ? "selected" : ""}>
          ${item.name} (${item.dataset_id})
        </option>
      `,
    )
    .join("");
}

function updateDatasetSummary() {
  const current = datasetCatalog.find((item) => item.dataset_id === datasetSelect.value);
  if (!current) return;
  datasetSummary.textContent = `${current.name} ｜ ${current.start_date} 至 ${current.end_date} ｜ ${current.symbol_count} symbols ｜ benchmark ${current.benchmark_symbol}`;
}

function renderSources(items) {
  sourceList.innerHTML = items
    .map(
      (item) => `
        <article class="source-item">
          <div class="source-head">
            <strong>${item.label}</strong>
            <span>${item.mode}</span>
          </div>
          <div class="source-text">${item.description}</div>
          <div class="source-datasets">${item.datasets.length ? item.datasets.join(", ") : "暂无已导入数据集"}</div>
        </article>
      `,
    )
    .join("");
}

function renderDatasetLibrary(items, selectedId) {
  datasetLibrary.innerHTML = items
    .map(
      (item) => `
        <button class="dataset-card ${item.dataset_id === selectedId ? "is-active" : ""}" data-dataset-id="${item.dataset_id}" type="button">
          <div class="dataset-card-head">
            <strong>${item.name}</strong>
            <span>${item.source_type}</span>
          </div>
          <div class="dataset-card-text">${item.start_date} 至 ${item.end_date}</div>
          <div class="dataset-card-text">${item.symbol_count} symbols / benchmark ${item.benchmark_symbol}</div>
        </button>
      `,
    )
    .join("");
}

function renderDatasetDetail(detail) {
  if (!detail) {
    datasetDetail.innerHTML = "";
    return;
  }
  const coverage = detail.coverage;
  const symbols = (detail.symbols || []).map((item) => item.symbol).join(", ");
  datasetDetail.innerHTML = `
    <div class="detail-grid">
      <div class="detail-item"><span>bars</span><strong>${coverage.bar_count}</strong></div>
      <div class="detail-item"><span>daily basic</span><strong>${coverage.daily_basic_count}</strong></div>
      <div class="detail-item"><span>financials</span><strong>${coverage.financial_count}</strong></div>
      <div class="detail-item"><span>events</span><strong>${coverage.event_count}</strong></div>
    </div>
    <div class="detail-note">symbols: ${symbols || "--"}</div>
  `;
}

function renderNotes(items) {
  notes.innerHTML = items.map((item) => `<div class="note-item">${item}</div>`).join("");
}

function renderMetrics(metrics) {
  const items = [
    ["总收益", pct(metrics.total_return)],
    ["年化收益", pct(metrics.annual_return)],
    ["最大回撤", pct(metrics.max_drawdown)],
    ["Sharpe", Number(metrics.sharpe_ratio || 0).toFixed(2)],
    ["基准收益", pct(metrics.benchmark_return)],
    ["超额收益", pct(metrics.excess_return)],
    ["交易次数", metrics.trade_count],
  ];
  metricGrid.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${value}</div>
        </div>
      `,
    )
    .join("");
}

function renderChart(points) {
  if (!points?.length) {
    chart.innerHTML = "";
    return;
  }
  const width = 900;
  const height = 320;
  const padding = 28;
  const values = points.flatMap((point) => [point.equity, point.benchmark_equity]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const buildPath = (key) =>
    points
      .map((point, index) => {
        const x = padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
        const y = height - padding - ((point[key] - min) / range) * (height - padding * 2);
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(" ");

  chart.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="18" class="chart-bg"></rect>
    <path d="${buildPath("equity")}" class="line strategy"></path>
    <path d="${buildPath("benchmark_equity")}" class="line benchmark"></path>
    <text x="${padding}" y="24" class="chart-label">Strategy</text>
    <text x="${padding + 90}" y="24" class="chart-label benchmark-label">Benchmark</text>
  `;
}

function renderDecisions(decisions) {
  if (!decisions?.length) {
    decisionList.innerHTML = `<div class="empty">暂无调仓记录。</div>`;
    return;
  }
  decisionList.innerHTML = decisions
    .map((decision) => {
      const selectedCount = Object.keys(decision.target_weights || {}).length;
      const cards = decision.research_cards
        .slice(0, 3)
        .map(
          (card) => `
            <div class="research-card">
              <div class="research-head">
                <strong>${card.symbol}</strong>
                <span class="research-badge action-${card.action}">${titleCase(card.action)}</span>
              </div>
              <div class="research-meta">${titleCase(card.stance)} · Score ${Number(card.score || 0).toFixed(2)} · Conf ${Number(card.confidence || 0).toFixed(2)}</div>
              <div class="research-text">${card.summary}</div>
            </div>
          `,
        )
        .join("");
      return `
        <article class="decision-item">
          <div class="decision-head">
            <div class="decision-title">
              <h3>${decision.decision_date} → ${decision.execution_date}</h3>
              <p>${decision.market_context.market_view}</p>
            </div>
            <div class="decision-pill">${titleCase(decision.market_context.regime_label)}</div>
          </div>
          <div class="decision-stats">
            <div class="decision-stat">
              <span>持仓数</span>
              <strong>${selectedCount}</strong>
            </div>
            <div class="decision-stat">
              <span>现金仓位</span>
              <strong>${pct(decision.cash_weight)}</strong>
            </div>
            <div class="decision-stat">
              <span>执行日</span>
              <strong>${decision.execution_date}</strong>
            </div>
          </div>
          <div class="decision-summary">${decision.summary}</div>
          <div class="decision-section-label">目标组合</div>
          <div class="decision-weights">${renderWeightChips(decision.target_weights, decision.cash_weight)}</div>
          <div class="decision-section-label">Top Research Cards</div>
          <div class="research-grid">${cards}</div>
        </article>
      `;
    })
    .join("");
}

function renderTrades(trades) {
  if (!trades?.length) {
    tradeBody.innerHTML = `<tr><td colspan="5" class="empty-row">暂无交易。</td></tr>`;
    return;
  }
  tradeBody.innerHTML = trades
    .map(
      (trade) => `
        <tr>
          <td>
            <div class="trade-primary">${trade.trade_date}</div>
            <div class="trade-secondary">执行</div>
          </td>
          <td>
            <div class="trade-primary">${trade.decision_date}</div>
            <div class="trade-secondary">生成信号</div>
          </td>
          <td><span class="symbol-chip">${trade.symbol}</span></td>
          <td><span class="trade-side trade-side-${String(trade.side || "").toLowerCase()}">${trade.side}</span></td>
          <td class="trade-notional">${money(trade.notional)}</td>
        </tr>
      `,
    )
    .join("");
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const rawText = await response.text();
  let payload = null;
  try {
    payload = rawText ? JSON.parse(rawText) : null;
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(payload?.error || payload?.message || `Request failed: ${response.status}`);
  }
  return payload;
}

async function bootstrap() {
  const payload = await request("/api/bootstrap");
  const selected = payload.datasets.find((item) => item.dataset_id === payload.default_config.dataset_id) || payload.datasets[0];
  renderDatasetOptions(payload.datasets, payload.default_config.dataset_id);
  renderSources(payload.data_sources || []);
  renderDatasetLibrary(payload.datasets, payload.default_config.dataset_id);
  datasetSummary.textContent = `${selected.name} ｜ ${selected.start_date} 至 ${selected.end_date} ｜ ${selected.symbol_count} symbols ｜ benchmark ${selected.benchmark_symbol}`;
  setDefaults(payload.default_config);
  const llmConfigured = payload.llm?.configured;
  const bootstrapNotes = [...payload.notes];
  bootstrapNotes.push(llmConfigured ? `LLM 已配置：${payload.llm.model}` : "LLM 未配置：运行回测前请设置 API Key。");
  renderNotes(bootstrapNotes);
  await loadDatasetDetail(payload.default_config.dataset_id);
}

async function loadDatasetDetail(datasetId) {
  const detail = await request(`/api/datasets/${datasetId}`);
  renderDatasetDetail(detail);
}

async function runBacktest(event) {
  event.preventDefault();
  runButton.disabled = true;
  runButton.textContent = "运行中...";
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());
  try {
    const result = await request("/api/backtests/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const curve = result.equity_curve || [];
    if (curve.length) {
      const first = curve[0].trade_date;
      const last = curve[curve.length - 1].trade_date;
      resultRange.textContent = `${first} 至 ${last}`;
    }
    renderMetrics(result.metrics);
    renderNotes(result.notes);
    renderChart(curve);
    renderDecisions(result.decisions);
    renderTrades(result.trades);
  } catch (error) {
    notes.innerHTML = `<div class="note-item error">回测失败：${error.message}</div>`;
  } finally {
    runButton.disabled = false;
    runButton.textContent = "运行回测";
  }
}

form.addEventListener("submit", runBacktest);
datasetSelect.addEventListener("change", async () => {
  updateDatasetSummary();
  renderDatasetLibrary(datasetCatalog, datasetSelect.value);
  await loadDatasetDetail(datasetSelect.value);
});
datasetLibrary.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-dataset-id]");
  if (!button) return;
  datasetSelect.value = button.dataset.datasetId;
  updateDatasetSummary();
  renderDatasetLibrary(datasetCatalog, datasetSelect.value);
  await loadDatasetDetail(datasetSelect.value);
});
bootstrap();
