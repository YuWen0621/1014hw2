/* ==========================================================================
   HW2: Student Performance Multiple Linear Regression & Feature Selection Explorer
   Pure Vanilla JavaScript + Chart.js 4.4 (GitHub Pages Ready)
   Dataset: Kaggle Student Performance Challenge (16,000 Samples, 17 Features)
   ========================================================================== */

// Application State
let currentAlgo = 'Pearson Correlation';
let currentK = 6; // Default to Global Sweet Spot (k = 6)
let rmseChartInstance = null;
let r2ChartInstance = null;
let actualVsPredChartInstance = null;

const ALGO_COLORS = {
  'Pearson Correlation': '#336640',         // Deep Forest / Sage Green
  'Forward SFS': '#ef4444',                 // Red
  'RFECV': '#2a9d8f',                       // Mint Teal
  'Exhaustive Feature Selection': '#7663a8', // Lavender Purple
  'LassoCV': '#d97706',                     // Warm Amber
  'Baseline OLS': '#457b9d'                 // Muted Slate Blue
};

const ALGO_DASH = {
  'Pearson Correlation': [],                // Solid
  'Forward SFS': [8, 4],                    // Long dash
  'RFECV': [4, 4],                          // Short dash
  'Exhaustive Feature Selection': [10, 3, 3, 3], // Dash-dot
  'LassoCV': [2, 3],                        // Dotted
  'Baseline OLS': [6, 3, 1, 3]              // Dash-dot-dot
};

const ALGO_POINT_STYLE = {
  'Pearson Correlation': 'circle',
  'Forward SFS': 'triangle',
  'RFECV': 'rect',
  'Exhaustive Feature Selection': 'star',
  'LassoCV': 'rectRot',
  'Baseline OLS': 'crossRot'
};

const ALL_ALGO_KEYS = [
  'Pearson Correlation',
  'Forward SFS',
  'RFECV',
  'Exhaustive Feature Selection',
  'LassoCV',
  'Baseline OLS'
];

let activeMethodsFilter = new Set(ALL_ALGO_KEYS);

// Helper: Get Current Dataset Definition
function getCurrentDataset() {
  if (window.DATASETS && window.DATASETS['student_performance']) {
    return window.DATASETS['student_performance'];
  }
  return null;
}

// Helper: Get Active Model Data for current (Algo, K)
function getActiveModelData() {
  if (!window.MODEL_DATA) return null;
  const algo = currentAlgo === 'ALL' ? 'Pearson Correlation' : currentAlgo;
  const models = window.MODEL_DATA[algo];
  if (!models || models.length === 0) return null;
  const idx = Math.min(Math.max(currentK - 1, 0), models.length - 1);
  return models[idx];
}

// DOM Initialization
document.addEventListener('DOMContentLoaded', () => {
  const ds = getCurrentDataset();
  if (ds) {
    window.MODEL_DATA = ds.model_data;
    window.METADATA = ds.metadata;
    window.FINAL_MODEL_COMPARISON = ds.final_model_comparison;
    currentK = ds.sweet_spot_k || 6;
  }

  setupEventListeners();
  renderKButtons();
  initCharts();
  renderLivePredictor();
  updateDashboard();
});

// Event Listeners Setup
function setupEventListeners() {
  // Algorithm selection tabs
  const tabs = document.querySelectorAll('#algoTabs .tab-btn');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentAlgo = btn.dataset.algo;
      updateDashboard();
    });
  });

  // Filter chips for Compare All mode
  initMethodFilterChips();
}

// Initialize Method Filter Chips for Compare All mode
function initMethodFilterChips() {
  const container = document.getElementById('methodFilterChips');
  if (!container) return;
  container.innerHTML = '';

  ALL_ALGO_KEYS.forEach(m => {
    const chip = document.createElement('button');
    chip.className = 'filter-chip active';
    chip.dataset.algo = m;
    chip.style.borderColor = ALGO_COLORS[m];
    chip.style.backgroundColor = ALGO_COLORS[m] + '18';
    chip.style.color = ALGO_COLORS[m];
    chip.innerHTML = `<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${ALGO_COLORS[m]}; margin-right:6px;"></span>${m}`;

    chip.addEventListener('click', () => {
      if (activeMethodsFilter.has(m)) {
        if (activeMethodsFilter.size > 1) {
          activeMethodsFilter.delete(m);
          chip.classList.remove('active');
          chip.style.opacity = '0.45';
        }
      } else {
        activeMethodsFilter.add(m);
        chip.classList.add('active');
        chip.style.opacity = '1.0';
      }
      updateCharts();
    });

    container.appendChild(chip);
  });

  const btnAll = document.getElementById('filterSelectAll');
  if (btnAll) {
    btnAll.addEventListener('click', () => {
      activeMethodsFilter = new Set(ALL_ALGO_KEYS);
      container.querySelectorAll('.filter-chip').forEach(c => {
        c.classList.add('active');
        c.style.opacity = '1.0';
      });
      updateCharts();
    });
  }

  const btnClear = document.getElementById('filterClearAll');
  if (btnClear) {
    btnClear.addEventListener('click', () => {
      activeMethodsFilter = new Set(['Pearson Correlation']);
      container.querySelectorAll('.filter-chip').forEach(c => {
        if (c.dataset.algo === 'Pearson Correlation') {
          c.classList.add('active');
          c.style.opacity = '1.0';
        } else {
          c.classList.remove('active');
          c.style.opacity = '0.45';
        }
      });
      updateCharts();
    });
  }
}

// Render dynamic k-selection buttons (k = 1..17)
function renderKButtons() {
  const container = document.getElementById('kSelectorGroup');
  const ds = getCurrentDataset();
  if (!container || !ds) return;
  container.innerHTML = '';

  const totalFeatures = ds.features.length; // 17

  for (let k = 1; k <= totalFeatures; k++) {
    const btn = document.createElement('button');
    btn.className = 'k-btn';
    btn.dataset.k = k;

    if (k === ds.sweet_spot_k) {
      btn.classList.add('sweet-spot');
      btn.innerHTML = `k = ${k} ★ Sweet Spot <span style="font-size: 11px; font-weight: 700;">(Min RMSE)</span>`;
    } else if (k === ds.overfitting_k) {
      btn.classList.add('overfitting');
      btn.innerHTML = `k = ${k} ⚠ Full Model <span style="font-size: 11px; font-weight: 700;">(Noise Features)</span>`;
    } else if (k === 1) {
      btn.innerHTML = `k = 1 <span style="font-size: 11px;">(Study Hours)</span>`;
    } else {
      btn.innerHTML = `k = ${k} <span style="font-size: 11px;">(${k} Predictors)</span>`;
    }

    if (k === currentK) {
      btn.classList.add('active');
    }

    btn.addEventListener('click', () => {
      container.querySelectorAll('.k-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentK = k;
      updateDashboard();
    });

    container.appendChild(btn);
  }
}

// Initialize All Chart.js Instances
function initCharts() {
  const ctxRMSE = document.getElementById('rmseChart');
  const ctxR2 = document.getElementById('r2Chart');
  const ctxActualVsPred = document.getElementById('actualVsPredChart');

  if (!ctxRMSE || !ctxR2) return;

  const chartFont = { family: "'Inter', sans-serif", size: 11 };

  // Common options
  const baseChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        labels: { font: chartFont, usePointStyle: true, boxWidth: 8, padding: 12 }
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.92)',
        titleFont: { family: "'Outfit', sans-serif", weight: 'bold', size: 12 },
        bodyFont: chartFont,
        padding: 10,
        cornerRadius: 8
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(226, 232, 240, 0.7)' },
        ticks: { font: chartFont, color: '#64748b' },
        title: { display: true, text: 'Number of Features (k)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' }
      },
      y: {
        grid: { color: 'rgba(226, 232, 240, 0.7)' },
        ticks: { font: chartFont, color: '#64748b' }
      }
    }
  };

  // 1. RMSE Chart
  const rmseOptions = JSON.parse(JSON.stringify(baseChartOptions));
  rmseOptions.scales.y.title = { display: true, text: 'Test RMSE (Lower is Better)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' };
  rmseOptions.scales.y.ticks.callback = (val) => Number(val).toFixed(2);
  rmseChartInstance = new Chart(ctxRMSE, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: rmseOptions
  });

  // 2. R2 Chart
  const r2Options = JSON.parse(JSON.stringify(baseChartOptions));
  r2Options.scales.y.title = { display: true, text: 'Test R² (Higher is Better)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' };
  r2Options.scales.y.min = 0.40;
  r2Options.scales.y.max = 1.05;
  r2Options.scales.y.ticks.callback = (val) => Number(val).toFixed(2);
  r2ChartInstance = new Chart(ctxR2, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: r2Options
  });

  // 3. Actual vs Predicted Chart with 95% Prediction Interval
  if (ctxActualVsPred) {
    actualVsPredChartInstance = new Chart(ctxActualVsPred, {
      type: 'scatter',
      data: { datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
          legend: {
            position: 'top',
            labels: { font: chartFont, usePointStyle: true, boxWidth: 8, padding: 12 }
          },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.94)',
            titleFont: { family: "'Outfit', sans-serif", weight: 'bold', size: 12 },
            bodyFont: chartFont,
            callbacks: {
              label: (ctx) => {
                const pt = ctx.raw;
                if (pt.label === 'Ideal Line') return `Ideal Fit: y = ŷ (${pt.x.toFixed(2)})`;
                if (pt.label === 'Upper 95% PI') return `Upper 95% PI: ${pt.y.toFixed(2)}`;
                if (pt.label === 'Lower 95% PI') return `Lower 95% PI: ${pt.y.toFixed(2)}`;
                return [
                  `Predicted (ŷ): ${pt.x.toFixed(2)} pts`,
                  `Actual (y): ${pt.y.toFixed(2)} pts`,
                  `Residual: ${(pt.y - pt.x >= 0 ? '+' : '') + (pt.y - pt.x).toFixed(4)} pts`,
                  `95% PI: [${(pt.pi_low !== undefined ? pt.pi_low.toFixed(2) : 'N/A')}, ${(pt.pi_up !== undefined ? pt.pi_up.toFixed(2) : 'N/A')}]`,
                  `Status: ${pt.inside ? '✓ Inside 95% PI' : '⚠ Outside PI'}`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(226, 232, 240, 0.7)' },
            ticks: { font: chartFont, color: '#64748b' },
            title: { display: true, text: 'Predicted Productivity Score (ŷ)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' }
          },
          y: {
            grid: { color: 'rgba(226, 232, 240, 0.7)' },
            ticks: { font: chartFont, color: '#64748b' },
            title: { display: true, text: 'Actual Productivity Score (y)', font: { family: "'Inter', sans-serif", weight: 'bold', size: 11.5 }, color: '#334155' }
          }
        }
      }
    });
  }
}

// Update Charts Data based on selection
function updateCharts() {
  if (!rmseChartInstance || !r2ChartInstance || !window.MODEL_DATA) return;
  const ds = getCurrentDataset();
  if (!ds) return;

  const labels = ds.k_labels || [];
  rmseChartInstance.data.labels = labels;
  r2ChartInstance.data.labels = labels;

  const rmseDatasets = [];
  const r2Datasets = [];

  const methodsToPlot = currentAlgo === 'ALL' ? ALL_ALGO_KEYS.filter(m => activeMethodsFilter.has(m)) : [currentAlgo];

  methodsToPlot.forEach(algo => {
    const algoData = window.MODEL_DATA[algo];
    if (!algoData) return;

    const testRMSE = algoData.map(d => d.rmse);
    const testR2 = algoData.map(d => d.r2);
    const cvRMSE = algoData.map(d => d.cv_rmse);
    const cvR2 = algoData.map(d => d.cv_r2);

    const color = ALGO_COLORS[algo] || '#4f46e5';
    const dash = ALGO_DASH[algo] || [];
    const ptStyle = ALGO_POINT_STYLE[algo] || 'circle';

    if (currentAlgo === 'ALL') {
      // Compare all mode: Show Test RMSE and Test R2
      rmseDatasets.push({
        label: `${algo}`,
        data: testRMSE,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2.2,
        borderDash: dash,
        pointStyle: ptStyle,
        pointRadius: 4.5,
        pointHoverRadius: 7,
        tension: 0.15
      });

      r2Datasets.push({
        label: `${algo}`,
        data: testR2,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2.2,
        borderDash: dash,
        pointStyle: ptStyle,
        pointRadius: 4.5,
        pointHoverRadius: 7,
        tension: 0.15
      });
    } else {
      // Single method mode: Show both CV and Test curves
      rmseDatasets.push({
        label: `${algo} (CV RMSE)`,
        data: cvRMSE,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2.4,
        pointStyle: ptStyle,
        pointRadius: 5.5,
        tension: 0.15
      });
      rmseDatasets.push({
        label: `${algo} (Test RMSE)`,
        data: testRMSE,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 1.6,
        borderDash: [5, 4],
        pointStyle: ptStyle,
        pointRadius: 4.5,
        tension: 0.15
      });

      r2Datasets.push({
        label: `${algo} (CV R²)`,
        data: cvR2,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2.4,
        pointStyle: ptStyle,
        pointRadius: 5.5,
        tension: 0.15
      });
      r2Datasets.push({
        label: `${algo} (Test R²)`,
        data: testR2,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 1.6,
        borderDash: [5, 4],
        pointStyle: ptStyle,
        pointRadius: 4.5,
        tension: 0.15
      });
    }
  });

  // Highlight Current Selected K on Charts
  const activeModel = getActiveModelData();
  if (activeModel) {
    const kIdx = activeModel.k - 1;
    const highlightDataset = {
      label: `Active Selection (k = ${activeModel.k})`,
      data: labels.map((_, i) => i === kIdx ? activeModel.rmse : null),
      borderColor: '#10b981',
      backgroundColor: '#10b981',
      pointRadius: 8,
      pointHoverRadius: 10,
      showLine: false,
      pointStyle: 'star'
    };
    rmseDatasets.push(highlightDataset);

    const highlightR2Dataset = {
      label: `Active Selection (k = ${activeModel.k})`,
      data: labels.map((_, i) => i === kIdx ? activeModel.r2 : null),
      borderColor: '#10b981',
      backgroundColor: '#10b981',
      pointRadius: 8,
      pointHoverRadius: 10,
      showLine: false,
      pointStyle: 'star'
    };
    r2Datasets.push(highlightR2Dataset);
  }

  rmseChartInstance.data.datasets = rmseDatasets;
  r2ChartInstance.data.datasets = r2Datasets;

  rmseChartInstance.update();
  r2ChartInstance.update();

  // Update Actual vs Predicted Chart
  updateActualVsPredChart();
}

// Update Actual vs Predicted Chart with 95% Prediction Interval
function updateActualVsPredChart() {
  if (!actualVsPredChartInstance) return;
  const ds = getCurrentDataset();
  const activeModel = getActiveModelData();
  if (!ds || !activeModel) return;

  const samplePoints = ds.test_scatter_sample || [];
  if (samplePoints.length === 0) return;

  // Find min and max for the 45-degree reference line
  let minVal = 0;
  let maxVal = 100;

  // Calculate prediction margin of error for current active model
  // Margin = t_crit * SE_pred
  const margin = (activeModel.mean_pi_width || 0.02) / 2;

  // Scatter points: (predicted, actual)
  const inlierPoints = [];
  const outlierPoints = [];

  samplePoints.forEach(p => {
    const pt = {
      x: p.predicted,
      y: p.actual,
      pi_low: p.predicted - margin,
      pi_up: p.predicted + margin,
      inside: Math.abs(p.actual - p.predicted) <= margin
    };
    if (pt.inside) {
      inlierPoints.push(pt);
    } else {
      outlierPoints.push(pt);
    }
  });

  // Reference lines for 45-degree and 95% PI envelope
  const linePoints = [
    { x: minVal, y: minVal, label: 'Ideal Line' },
    { x: maxVal, y: maxVal, label: 'Ideal Line' }
  ];

  const upperPIPoints = [
    { x: minVal, y: minVal + margin, label: 'Upper 95% PI' },
    { x: maxVal, y: maxVal + margin, label: 'Upper 95% PI' }
  ];

  const lowerPIPoints = [
    { x: minVal, y: Math.max(0, minVal - margin), label: 'Lower 95% PI' },
    { x: maxVal, y: maxVal - margin, label: 'Lower 95% PI' }
  ];

  actualVsPredChartInstance.data.datasets = [
    {
      label: 'Test Observations (Inside 95% PI)',
      data: inlierPoints,
      backgroundColor: 'rgba(59, 130, 246, 0.65)',
      borderColor: '#2563eb',
      borderWidth: 1,
      pointRadius: 4,
      pointHoverRadius: 6
    },
    {
      label: 'Ideal Fit (y = ŷ)',
      data: linePoints,
      type: 'line',
      borderColor: '#ef4444',
      borderWidth: 2,
      borderDash: [6, 4],
      fill: false,
      pointRadius: 0
    },
    {
      label: 'Upper 95% PI Bound (ŷ + t·SE)',
      data: upperPIPoints,
      type: 'line',
      borderColor: '#10b981',
      borderWidth: 1.5,
      borderDash: [3, 3],
      fill: '+1',
      backgroundColor: 'rgba(16, 185, 129, 0.15)',
      pointRadius: 0
    },
    {
      label: 'Lower 95% PI Bound (ŷ - t·SE)',
      data: lowerPIPoints,
      type: 'line',
      borderColor: '#10b981',
      borderWidth: 1.5,
      borderDash: [3, 3],
      fill: false,
      pointRadius: 0
    }
  ];

  if (outlierPoints.length > 0) {
    actualVsPredChartInstance.data.datasets.push({
      label: 'Test Observations (Outside PI)',
      data: outlierPoints,
      backgroundColor: 'rgba(239, 68, 68, 0.85)',
      borderColor: '#dc2626',
      borderWidth: 1.2,
      pointRadius: 5,
      pointHoverRadius: 7
    });
  }

  actualVsPredChartInstance.update();

  const piBadge = document.getElementById('piCoverageBadge');
  if (piBadge) {
    piBadge.innerHTML = `<i class="fa-solid fa-check-double"></i> ${activeModel.pi_coverage.toFixed(1)}% Test Set PI Coverage`;
  }
}

// Render dynamic Live Predictor inputs based on active model's features
function renderLivePredictor() {
  const ds = getCurrentDataset();
  const activeModel = getActiveModelData();
  const container = document.getElementById('livePredictorInputs');
  const titleEl = document.getElementById('livePredictorTitle');
  const subtitleEl = document.getElementById('livePredictorSubtitle');

  if (!container || !ds || !activeModel) return;
  container.innerHTML = '';

  if (titleEl) titleEl.textContent = 'Live Student Productivity Predictor';
  if (subtitleEl) subtitleEl.textContent = `Real-time prediction with statistically calculated 95% Prediction Interval for active model (k = ${activeModel.k}).`;

  const activeFeatures = new Set(activeModel.features);

  // Filter and display input controls for active quantitative features
  ds.predictor_features.forEach(p => {
    if (!activeFeatures.has(p.feature)) return;

    const group = document.createElement('div');
    group.className = 'input-group';

    const unitStr = p.unit ? ` (${p.unit})` : '';

    group.innerHTML = `
      <div class="input-label-row">
        <span>${p.name}${unitStr}</span>
        <span id="${p.labelId}" style="color: var(--primary); font-family: 'Fira Code', monospace; font-weight: 600;">${p.default}${p.unit ? ' ' + p.unit : ''}</span>
      </div>
      <input type="number" id="${p.id}" class="input-field" value="${p.default}" min="${p.min}" max="${p.max}" step="${p.step}">
      <input type="range" id="${p.sliderId}" class="range-slider" value="${p.default}" min="${p.min}" max="${p.max}" step="${p.step}">
    `;
    container.appendChild(group);

    const numInput = group.querySelector(`#${p.id}`);
    const sliderInput = group.querySelector(`#${p.sliderId}`);
    const lbl = group.querySelector(`#${p.labelId}`);

    const sync = (val) => {
      numInput.value = val;
      sliderInput.value = val;
      if (lbl) {
        lbl.textContent = val + (p.unit ? ' ' + p.unit : '');
      }
      calculateLivePrediction();
    };

    numInput.addEventListener('input', (e) => sync(e.target.value));
    sliderInput.addEventListener('input', (e) => sync(e.target.value));
  });

  // If gender dummy features are active, render gender select
  const hasGenderMale = activeFeatures.has('gender_Male');
  const hasGenderOther = activeFeatures.has('gender_Other');

  if (hasGenderMale || hasGenderOther) {
    const catGroup = document.createElement('div');
    catGroup.className = 'input-group';
    catGroup.id = 'inputGroupGender';

    catGroup.innerHTML = `
      <div class="input-label-row">
        <span>Gender (Categorical)</span>
      </div>
      <select id="selectGender" class="input-field">
        <option value="Female" selected>Female (Reference Level)</option>
        <option value="Male">Male</option>
        <option value="Other">Other</option>
      </select>
    `;
    container.appendChild(catGroup);

    const sel = catGroup.querySelector('#selectGender');
    if (sel) {
      sel.addEventListener('change', calculateLivePrediction);
    }
  }

  calculateLivePrediction();
}

// Live Interactive Predictor Calculation with exact 95% Prediction Interval
function calculateLivePrediction() {
  const data = getActiveModelData();
  const ds = getCurrentDataset();
  if (!data || !ds) return;

  const activeFeatures = data.features;
  const pCount = activeFeatures.length;

  // Vector x0 = [1, x1, x2, ..., xp]
  const x0 = [1.0];
  let pointPred = 0.0;
  const breakdown = [];

  const interceptRow = data.coefficients.find(c => c.feature.includes('Intercept'));
  const b0 = interceptRow ? interceptRow.coef : 0.0;
  pointPred += b0;
  breakdown.push(`Intercept (β₀): ${b0.toFixed(4)}`);

  // Gender selection
  const selectGender = document.getElementById('selectGender');
  const genderVal = selectGender ? selectGender.value : 'Female';

  data.coefficients.forEach(c => {
    if (c.feature.includes('Intercept')) return;

    let featVal = 0.0;

    if (c.feature === 'gender_Male') {
      featVal = (genderVal === 'Male') ? 1.0 : 0.0;
    } else if (c.feature === 'gender_Other') {
      featVal = (genderVal === 'Other') ? 1.0 : 0.0;
    } else {
      // Find matching input control
      const meta = ds.predictor_features.find(p => p.feature === c.feature);
      if (meta) {
        const inputEl = document.getElementById(meta.id);
        featVal = inputEl ? parseFloat(inputEl.value) || 0.0 : meta.default;
      }
    }

    x0.push(featVal);
    const contribution = c.coef * featVal;
    pointPred += contribution;
    const sign = contribution >= 0 ? '+' : '';
    breakdown.push(`${c.feature}: ${sign}${contribution.toFixed(4)}`);
  });

  // Calculate Leverage and 95% Prediction Interval:
  // leverage = x0^T * (X^T X)^-1 * x0
  // SE_pred = sqrt( sigma2 * (1 + leverage) )
  // Margin = t_crit * SE_pred
  let leverage = 0.0;
  const invXtX = data.inv_XtX;

  if (invXtX && invXtX.length === x0.length) {
    for (let i = 0; i < x0.length; i++) {
      for (let j = 0; j < x0.length; j++) {
        leverage += x0[i] * invXtX[i][j] * x0[j];
      }
    }
  }

  const s2 = data.sigma2 || 0.0001;
  const tCrit = data.t_critical || 1.960;
  const sePred = Math.sqrt(s2 * (1.0 + Math.max(0, leverage)));
  const margin = tCrit * sePred;

  const piLower = Math.max(0, pointPred - margin);
  const piUpper = Math.min(100, pointPred + margin);

  // Update Prediction Output Elements
  const outputEl = document.getElementById('predProfitOutput');
  const piOutputEl = document.getElementById('predPIOutput');
  const noteEl = document.getElementById('predFormulaNote');

  if (outputEl) {
    outputEl.textContent = `${pointPred.toFixed(2)} pts`;
  }

  if (piOutputEl) {
    piOutputEl.innerHTML = `
      <div style="font-size: 14px; font-weight: 700; color: #047857;">
        <i class="fa-solid fa-shield-halved"></i> 95% Prediction Interval: 
        <span style="font-family: 'Fira Code', monospace; color: #065f46;">[${piLower.toFixed(2)}, ${piUpper.toFixed(2)}] pts</span>
        <span style="font-size: 12px; color: #059669; font-weight: 600;"> (±${margin.toFixed(4)})</span>
      </div>
      <div style="font-size: 11px; color: #64748b; margin-top: 2px;">
        Statistical SE(pred) = ${sePred.toFixed(6)} • Leverage h = ${leverage.toFixed(6)} • t_crit = ${tCrit.toFixed(3)}
      </div>
    `;
  }

  if (noteEl) {
    noteEl.innerHTML = `
      <strong>Active Model:</strong> ${currentAlgo} (k = ${currentK})<br>
      <span style="color: #64748b; font-size: 11.5px;">Fitted Decomposition: ${breakdown.join(' | ')}</span>
    `;
  }
}

// Update Equation Box
function updateEquation(coefficients) {
  const box = document.getElementById('equationBox');
  if (!box || !coefficients) return;

  const intercept = coefficients.find(c => c.feature.includes('Intercept'));
  const parts = [];

  if (intercept) {
    const formattedIntercept = intercept.coef.toFixed(4);
    parts.push(`<span class="eq-intercept">${formattedIntercept}</span>`);
  }

  coefficients.forEach(c => {
    if (c.feature.includes('Intercept')) return;
    const sign = c.coef >= 0 ? '<span class="eq-pos">+</span>' : '<span class="eq-neg">-</span>';
    const absVal = Math.abs(c.coef).toFixed(4);
    parts.push(`${sign} ${absVal} × <span class="eq-var">(${c.feature})</span>`);
  });

  box.innerHTML = `<strong>Productivity Score</strong> = ${parts.join(' ')}`;
}

// Update Coefficients Table
function updateCoefficientsTable(coefficients) {
  const tbody = document.getElementById('coefTableBody');
  if (!tbody || !coefficients) return;
  tbody.innerHTML = '';

  coefficients.forEach(c => {
    const tr = document.createElement('tr');
    const isIntercept = c.feature.includes('Intercept');

    let pBadge = '';
    if (c.pvalue < 0.05) {
      const pText = c.pvalue < 0.0001 ? '< 0.0001' : c.pvalue.toFixed(4);
      pBadge = `<span class="badge-tag badge-green">${pText} <i class="fa-solid fa-check"></i></span>`;
    } else {
      pBadge = `<span class="badge-tag badge-amber">${c.pvalue.toFixed(4)} (n.s.)</span>`;
    }

    let vifDisplay = '<span style="color: var(--slate-400);">N/A</span>';
    if (!isIntercept && c.vif !== null && c.vif !== undefined) {
      if (c.vif < 5.0) {
        vifDisplay = `<span class="badge-tag badge-green">${c.vif.toFixed(2)} (Safe)</span>`;
      } else {
        vifDisplay = `<span class="badge-tag badge-amber">${c.vif.toFixed(2)} (Collinear)</span>`;
      }
    }

    const ciStr = `[${c.ci_lower.toFixed(4)}, ${c.ci_upper.toFixed(4)}]`;

    tr.innerHTML = `
      <td style="font-weight: 600; color: var(--slate-900);">${c.feature}</td>
      <td style="font-family: 'Fira Code', monospace; font-weight: 600;">${c.coef.toFixed(4)}</td>
      <td style="font-family: 'Fira Code', monospace; color: var(--slate-600);">${c.stderr.toFixed(6)}</td>
      <td style="font-family: 'Fira Code', monospace; font-weight: 600;">${c.tstat.toFixed(1)}</td>
      <td>${pBadge}</td>
      <td style="font-family: 'Fira Code', monospace; font-size: 12px; color: var(--slate-600);">${ciStr}</td>
      <td>${vifDisplay}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Update Residuals and Outliers Table
function updateOutliersTable(outliers) {
  const tbody = document.getElementById('outliersTableBody');
  if (!tbody || !outliers) return;
  tbody.innerHTML = '';

  outliers.forEach(o => {
    const tr = document.createElement('tr');
    const isHigh = o.abs_error > 0.01;
    const statusBadge = isHigh
      ? `<span class="badge-tag badge-amber"><span class="status-dot dot-amber"></span> High Residual</span>`
      : `<span class="badge-tag badge-green"><span class="status-dot dot-green"></span> Normal</span>`;

    const piStatus = o.inside_pi
      ? `<span class="badge-tag badge-green"><i class="fa-solid fa-check"></i> Inside 95% PI</span>`
      : `<span class="badge-tag badge-amber"><i class="fa-solid fa-xmark"></i> Outside PI</span>`;

    tr.innerHTML = `
      <td style="font-weight: 700;">#${o.rank}</td>
      <td>Student #${o.index}</td>
      <td style="font-family: 'Fira Code', monospace;">${o.actual.toFixed(2)} pts</td>
      <td style="font-family: 'Fira Code', monospace;">${o.predicted.toFixed(4)} pts</td>
      <td style="color: ${o.residual >= 0 ? '#059669' : '#dc2626'}; font-weight: 600; font-family: 'Fira Code', monospace;">${o.residual >= 0 ? '+' : ''}${o.residual.toFixed(4)}</td>
      <td style="font-family: 'Fira Code', monospace;">${o.abs_error.toFixed(4)}</td>
      <td style="font-family: 'Fira Code', monospace; font-weight: 600;">${o.pct_error.toFixed(4)}%</td>
      <td>${piStatus}</td>
    `;
    tbody.appendChild(tr);
  });
}

// Update Full Dashboard
function updateDashboard() {
  const data = getActiveModelData();
  const ds = getCurrentDataset();
  if (!data || !ds) return;

  // 1. KPI Cards
  const valRMSE = document.getElementById('valRMSE');
  const valR2 = document.getElementById('valR2');
  const valAdjR2 = document.getElementById('valAdjR2');
  const valMAE = document.getElementById('valMAE');
  const valFStat = document.getElementById('valFStat');
  const valCVRMSE = document.getElementById('valCVRMSE');
  const r2Note = document.getElementById('r2MetricNote');

  if (valRMSE) valRMSE.textContent = data.rmse.toFixed(4);
  if (valR2) valR2.textContent = data.r2.toFixed(4);
  if (valAdjR2) valAdjR2.textContent = data.adj_r2.toFixed(4);
  if (valMAE) valMAE.textContent = data.mae.toFixed(4);
  if (valFStat) valFStat.textContent = data.f_stat > 1e6 ? '> 10⁶' : data.f_stat.toFixed(1);
  if (valCVRMSE) valCVRMSE.textContent = data.cv_rmse.toFixed(4);

  if (r2Note) {
    r2Note.textContent = `Explains ${(data.r2 * 100).toFixed(2)}% of productivity score variance`;
  }

  // 2. Badges
  const rmseBadge = document.getElementById('rmseBadge');
  if (rmseBadge) {
    if (currentK === ds.sweet_spot_k) {
      rmseBadge.className = 'badge-tag badge-green';
      rmseBadge.innerHTML = '<i class="fa-solid fa-award"></i> Optimal Sweet Spot';
    } else if (currentK === ds.overfitting_k) {
      rmseBadge.className = 'badge-tag badge-amber';
      rmseBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Full Model';
    } else {
      rmseBadge.className = 'badge-tag badge-blue';
      rmseBadge.innerHTML = `k = ${currentK}`;
    }
  }

  // 3. Active Feature Chips
  const countEl = document.getElementById('featuresCount');
  if (countEl) countEl.textContent = data.features.length;

  const chipsContainer = document.getElementById('featuresListChips');
  if (chipsContainer) {
    chipsContainer.innerHTML = '';
    data.features.forEach(feat => {
      const chip = document.createElement('span');
      chip.className = 'feature-chip';
      chip.innerHTML = `<i class="fa-solid fa-check" style="color: #10b981; font-size: 10px;"></i> ${feat}`;
      chipsContainer.appendChild(chip);
    });
  }

  // 4. Parsimony Explanation Notice
  const notice = document.getElementById('parsimonyNotice');
  if (notice) {
    if (currentK === ds.sweet_spot_k) {
      notice.className = 'badge-tag badge-green';
      notice.innerHTML = `<i class="fa-solid fa-star"></i> Occam's Razor: 6 core predictors achieve near-zero RMSE (${data.rmse.toFixed(4)}) and perfect generalization (CV RMSE: ${data.cv_rmse.toFixed(4)}).`;
    } else if (currentK > ds.sweet_spot_k) {
      notice.className = 'badge-tag badge-amber';
      notice.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Redundant Features (${currentK} predictors): Adding noise features does not reduce RMSE and consumes degrees of freedom.`;
    } else {
      notice.className = 'badge-tag badge-blue';
      notice.innerHTML = `<i class="fa-solid fa-circle-info"></i> Underfitting (${currentK} predictors): Missing key lifestyle and study predictors; RMSE is elevated (${data.rmse.toFixed(2)}).`;
    }
  }

  // 5. Update Equation Box
  updateEquation(data.coefficients);

  // 6. Update Coefficients Table
  updateCoefficientsTable(data.coefficients);

  // 7. Update Outliers Table
  updateOutliersTable(data.outliers);

  // 8. Update Method Filter Panel Visibility
  const filterPanel = document.getElementById('methodFilterPanel');
  if (filterPanel) {
    filterPanel.style.display = (currentAlgo === 'ALL') ? 'block' : 'none';
  }

  // 9. Re-render live predictor controls for active features
  renderLivePredictor();

  // 10. Update Charts
  updateCharts();
}

// Expose updateDashboard to window
window.updateDashboard = updateDashboard;
