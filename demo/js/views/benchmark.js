/**
 * Benchmark View - Linear Algebra Performance Suite
 */

import { benchmarkDefinitions, libraries } from '../benchmark.js';

// Application state for benchmark runs
const state = {
  benchmarks: {},
  wasmReady: false,
};

function formatCode(rawCode) {
  return rawCode
    .replace(/\b(const|let|var|for|function|return|new)\b/g, '<span class="code-keyword">$1</span>')
    .replace(/\b(createMatrix|createSparsePoissonMatrix|createCscSparseProblem|createVector|matMul|mmul|multiply|dot|mul|inverse|inv|svd|solve|solveComplexSystem|SparseLU|SimplicialCholesky)\b/g, '<span class="code-function">$1</span>')
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
      <button class="run-all-btn" data-bench-id="${bench.id}">
        RUN ALL
      </button>
    </div>

    <!-- Results Table -->
    <div class="results-table-wrapper">
      <table class="results-table">
        <thead>
          <tr>
            ${bench.supportedLibs.map(libKey => `
              <th>${libraries[libKey].name}</th>
            `).join('')}
          </tr>
        </thead>
        <tbody>
          <tr>
            ${bench.supportedLibs.map(libKey => `
              <td id="result_${bench.id}_${libKey}" class="result-cell">-</td>
            `).join('')}
          </tr>
        </tbody>
      </table>
    </div>
  `;

  // Attach tab switching events
  const tabBtns = card.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const libKey = btn.getAttribute('data-lib');
      state.benchmarks[bench.id].activeLib = libKey;

      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const codeArea = card.querySelector(`#code_${bench.id}`);
      codeArea.innerHTML = formatCode(bench.codes[libKey]);
    });
  });

  // Attach single run button event
  const runSingleBtn = card.querySelector('.run-single-btn');
  runSingleBtn.addEventListener('click', async () => {
    const activeLib = state.benchmarks[bench.id].activeLib;
    await runSingleBenchmark(bench, activeLib);
  });

  // Attach RUN ALL button event
  const runAllBtn = card.querySelector('.run-all-btn');
  runAllBtn.addEventListener('click', async () => {
    await runAllBenchmarks(bench, runAllBtn);
  });

  return card;
}

async function runSingleBenchmark(bench, libKey) {
  const sizeInput = document.getElementById(`size_${bench.id}`);
  const iterInput = document.getElementById(`iter_${bench.id}`);
  const resultCell = document.getElementById(`result_${bench.id}_${libKey}`);

  const size = parseInt(sizeInput.value, 10);
  const iterations = parseInt(iterInput.value, 10);

  resultCell.textContent = '...';
  resultCell.className = 'result-cell running';

  // Allow browser to render loading state
  await delay(16);

  try {
    const elapsed = bench.run(libKey, size, iterations);
    const rounded = Math.round(elapsed);
    state.benchmarks[bench.id].results[libKey] = rounded;

    resultCell.textContent = `${rounded}ms`;
    resultCell.className = 'result-cell';
  } catch (err) {
    console.error(`Error in ${bench.id} for ${libKey}:`, err);
    resultCell.textContent = 'Error';
    resultCell.className = 'result-cell error';
    state.benchmarks[bench.id].results[libKey] = Infinity;
  }

  updateFastestHighlight(bench);
}

async function runAllBenchmarks(bench, runAllBtn) {
  const originalText = runAllBtn.textContent;
  runAllBtn.disabled = true;
  runAllBtn.innerHTML = 'RUNNING...';

  // Reset current results
  bench.supportedLibs.forEach(libKey => {
    const cell = document.getElementById(`result_${bench.id}_${libKey}`);
    cell.textContent = '-';
    cell.className = 'result-cell';
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

export function renderBenchmarkView(container) {
  container.innerHTML = `
    <div class="benchmark-view">
      <section class="intro-section">
        <h1 class="page-title">Benchmarks of linear algebra javascript libraries</h1>
        <p class="page-subtitle">These tables compare the performances of different javascript linear algebra libraries</p>
        <div class="status-badge" id="wasmStatus">
          <span class="status-dot"></span>
          <span class="status-text">Loading WebAssembly engine...</span>
        </div>
      </section>

      <!-- Benchmarks List Container -->
      <div class="benchmarks-list" id="benchmarksContainer"></div>
    </div>
  `;

  const benchmarksContainer = container.querySelector('#benchmarksContainer');
  const wasmStatus = container.querySelector('#wasmStatus');

  // Render cards
  benchmarkDefinitions.forEach(bench => {
    state.benchmarks[bench.id] = {
      activeLib: bench.supportedLibs[0],
      results: {}
    };
    benchmarksContainer.appendChild(renderBenchmarkCard(bench));
  });

  // Check engines
  updateWasmStatus(wasmStatus);
}

async function updateWasmStatus(wasmStatus) {
  try {
    const promises = [];
    if (window.eig?.ready) promises.push(window.eig.ready);
    if (window.openBlasReady) promises.push(window.openBlasReady);
    if (window.umfpackReady) promises.push(window.umfpackReady);

    await Promise.all(promises);
    state.wasmReady = true;
    if (wasmStatus) {
      wasmStatus.className = 'status-badge ready';
      wasmStatus.querySelector('.status-text').textContent = 'WebAssembly Engines Ready (Eigen.js, OpenBLAS, SuiteSparse)';
    }
  } catch (e) {
    if (wasmStatus) {
      wasmStatus.className = 'status-badge ready';
      wasmStatus.querySelector('.status-text').textContent = 'Eigen.js WebAssembly Ready';
    }
  }
}
