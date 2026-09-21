import { benchmarkDefinitions, libraries } from './benchmark.js';

// Application state
const state = {
  benchmarks: {},
  wasmReady: false,
};

function formatCode(rawCode) {
  return rawCode
    .replace(/\b(const|let|var|for|function|return|new)\b/g, '<span class="code-keyword">$1</span>')
    .replace(/\b(createMatrix|matMul|mmul|multiply|dot|mul|inverse|inv|svd)\b/g, '<span class="code-function">$1</span>')
    .replace(/('[\w]+'|"[\w]+")/g, '<span class="code-string">$1</span>');
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function renderBenchmarkCard(bench) {
  const card = document.createElement('div');
  card.className = 'benchmark-card';
  card.id = `card_${bench.id}`;

  const currentTab = bench.supportedLibs[0];

  card.innerHTML = `
    <div class="benchmark-header">
      <h2 class="benchmark-title">${bench.name}</h2>
    </div>

    <!-- Tabs Bar -->
    <div class="tabs-bar" role="tablist">
      ${bench.supportedLibs.map((libKey, index) => `
        <button class="tab-btn ${index === 0 ? 'active' : ''}" data-bench-id="${bench.id}" data-lib="${libKey}">
          ${libraries[libKey].name}
        </button>
      `).join('')}
    </div>

    <!-- Code Display Area -->
    <div class="code-wrapper">
      <pre class="code-area" id="code_${bench.id}">${formatCode(bench.codes[currentTab])}</pre>
      <button class="run-single-btn" data-bench-id="${bench.id}" title="Run this library benchmark" aria-label="Run library benchmark">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
      </button>
    </div>

    <!-- Controls Row -->
    <div class="controls-row">
      <div class="input-field">
        <label for="size_${bench.id}">Matrix size</label>
        <input type="number" id="size_${bench.id}" value="${bench.params.size}" min="2" max="1000" step="10">
      </div>
      <div class="input-field">
        <label for="iter_${bench.id}">Iteration count</label>
        <input type="number" id="iter_${bench.id}" value="${bench.params.iterations}" min="1" max="10000" step="10">
      </div>
      <button class="run-all-btn" id="runAll_${bench.id}" data-bench-id="${bench.id}">
        <span class="btn-text">RUN ALL</span>
      </button>
    </div>

    <!-- Results Table -->
    <div class="results-table-container">
      <table class="results-table">
        <thead>
          <tr>
            ${bench.supportedLibs.map(libKey => `<th>${libraries[libKey].name}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
          <tr>
            ${bench.supportedLibs.map(libKey => `<td id="result_${bench.id}_${libKey}">-</td>`).join('')}
          </tr>
        </tbody>
      </table>
    </div>
  `;

  // Attach Tab Click Events
  const tabBtns = card.querySelectorAll('.tab-btn');
  const codeArea = card.querySelector(`#code_${bench.id}`);
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const selectedLib = btn.getAttribute('data-lib');
      state.benchmarks[bench.id].activeLib = selectedLib;
      codeArea.innerHTML = formatCode(bench.codes[selectedLib]);
    });
  });

  // Attach Run All Event
  const runAllBtn = card.querySelector(`#runAll_${bench.id}`);
  runAllBtn.addEventListener('click', () => runAllBenchmarks(bench));

  // Attach Run Single Event
  const runSingleBtn = card.querySelector('.run-single-btn');
  runSingleBtn.addEventListener('click', () => {
    const activeLib = state.benchmarks[bench.id].activeLib;
    runSingleBenchmark(bench, activeLib);
  });

  return card;
}

async function runSingleBenchmark(bench, libKey) {
  const resultCell = document.getElementById(`result_${bench.id}_${libKey}`);
  const sizeInput = document.getElementById(`size_${bench.id}`);
  const iterInput = document.getElementById(`iter_${bench.id}`);

  const size = parseInt(sizeInput.value, 10) || bench.params.size;
  const iterations = parseInt(iterInput.value, 10) || bench.params.iterations;

  resultCell.innerHTML = `<span class="spinner"></span>`;
  resultCell.className = 'running';

  await delay(30);

  try {
    const duration = bench.run(libKey, size, iterations);
    resultCell.className = '';
    resultCell.textContent = `${Math.round(duration)}ms`;
    state.benchmarks[bench.id].results[libKey] = duration;
    updateFastestHighlight(bench);
  } catch (err) {
    console.error(`Error executing ${bench.name} on ${libKey}:`, err);
    resultCell.className = '';
    resultCell.textContent = 'Error';
  }
}

async function runAllBenchmarks(bench) {
  const runAllBtn = document.getElementById(`runAll_${bench.id}`);
  const originalText = runAllBtn.innerHTML;

  runAllBtn.disabled = true;
  runAllBtn.innerHTML = `<span class="spinner"></span> Running...`;

  // Clear previous results
  bench.supportedLibs.forEach(libKey => {
    const cell = document.getElementById(`result_${bench.id}_${libKey}`);
    cell.textContent = '-';
    cell.className = '';
    state.benchmarks[bench.id].results[libKey] = null;
  });

  for (const libKey of bench.supportedLibs) {
    await runSingleBenchmark(bench, libKey);
    await delay(30);
  }

  runAllBtn.disabled = false;
  runAllBtn.innerHTML = originalText;
}

function updateFastestHighlight(bench) {
  const results = state.benchmarks[bench.id].results;
  const validTimes = Object.entries(results).filter(([_, time]) => typeof time === 'number' && !isNaN(time));

  if (validTimes.length < 2) return;

  const minTime = Math.min(...validTimes.map(([_, t]) => t));

  bench.supportedLibs.forEach(libKey => {
    const cell = document.getElementById(`result_${bench.id}_${libKey}`);
    const time = results[libKey];
    if (time === minTime) {
      cell.classList.add('fastest');
      if (!cell.textContent.includes('✓')) {
        cell.textContent += ' ✓';
      }
    } else {
      cell.classList.remove('fastest');
    }
  });
}

// Initialization
async function init() {
  const container = document.getElementById('benchmarksContainer');
  const wasmStatus = document.getElementById('wasmStatus');

  // Render cards
  benchmarkDefinitions.forEach(bench => {
    state.benchmarks[bench.id] = {
      activeLib: bench.supportedLibs[0],
      results: {}
    };
    container.appendChild(renderBenchmarkCard(bench));
  });

  // Check WebAssembly Engines Ready
  try {
    const promises = [];
    if (window.eig?.ready) promises.push(window.eig.ready);
    if (window.openBlasReady) promises.push(window.openBlasReady);
    if (window.umfpackReady) promises.push(window.umfpackReady);

    await Promise.all(promises);
    state.wasmReady = true;
    wasmStatus.className = 'status-badge ready';
    wasmStatus.querySelector('.status-text').textContent = 'WebAssembly Engines Ready (Eigen.js, OpenBLAS, SuiteSparse)';
  } catch (e) {
    console.error('Failed to initialize WebAssembly engines:', e);
    wasmStatus.className = 'status-badge ready';
    wasmStatus.querySelector('.status-text').textContent = 'Eigen.js WebAssembly Ready';
  }
}

// Start app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
