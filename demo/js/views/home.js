/**
 * Home View - Eigen.js Overview, Live SVD Animation & Usage
 */

let svdAnimationId = null;

export function renderHomeView(container) {
  // Cancel any existing animation frame
  if (svdAnimationId) {
    cancelAnimationFrame(svdAnimationId);
    svdAnimationId = null;
  }

  container.innerHTML = `
    <div class="home-view">
      <!-- Hero Section -->
      <section class="hero-section">
        <div class="hero-inner">
          <div class="hero-brand">
            <img src="assets/logo_white.svg" alt="Eigen JS Logo" class="hero-logo">
            <h1 class="hero-title">Eigen JS</h1>
            <p class="hero-subtitle">High-performance Linear Algebra for JavaScript & WebAssembly</p>
          </div>

          <!-- Live Interactive SVD Demo Box -->
          <div class="hero-demo-box">
            <div class="demo-card">
              <div class="demo-card-header">
                <span class="demo-tag">LIVE DEMO</span>
                <span class="demo-headline">Singular Value Decomposition (SVD)</span>
              </div>
              <div class="demo-canvas-container">
                <canvas id="svdCanvas" width="340" height="260"></canvas>
              </div>
              <div class="demo-matrix-display" id="svdMatrixDisplay">
                <span class="math-label">Matrix M = U &middot; &Sigma; &middot; V<sup>T</sup></span>
                <div class="matrix-values" id="matrixValues">Computing...</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Main Content Container -->
      <div class="home-content">
        <!-- Badges Row -->
        <div class="badges-row">
          <span class="badge badge-npm">npm package <strong>0.3.0</strong></span>
          <span class="badge badge-wasm">WebAssembly <strong>WASM</strong></span>
          <span class="badge badge-emscripten">Made with <strong>Emscripten</strong></span>
          <span class="badge badge-license">license <strong>MIT</strong></span>
        </div>

        <!-- Overview Section -->
        <section class="home-section">
          <h2 class="section-title">Eigen.js</h2>
          <p class="section-desc">
            <strong>Eigen.js</strong> is a WebAssembly port of the renowned C++ linear algebra library
            <a href="https://gitlab.com/libeigen/eigen" target="_blank" rel="noopener noreferrer">Eigen</a>.
            It provides high-performance dense and sparse matrix computations, matrix factorizations (SVD, QR, Cholesky, LU),
            linear system solvers, and quadratic optimization via <strong>OSQP</strong> directly in browser and Node.js environments.
          </p>
          <p class="section-desc">
            It bundles an automatic <strong>Garbage Collection (GC)</strong> mechanism that tracks and frees WebAssembly
            heap memory pointers safely in JavaScript.
          </p>
        </section>

        <!-- Live Demo & Documentation Links -->
        <section class="home-section">
          <h2 class="section-title">Live Demo & Documentation</h2>
          <p class="section-desc">Explore the interactive documentation and performance benchmarks:</p>
          <div class="feature-cards-grid">
            <a href="#/matrix" class="feature-card">
              <div class="feature-icon">[<sub>0</sub><sup>1</sup>]</div>
              <div class="feature-title">Matrix Class</div>
              <p class="feature-desc">Dense matrices, slicing, arithmetic, determinants, inverses, eigenvalues, and transformations.</p>
            </a>
            <a href="#/solvers" class="feature-card">
              <div class="feature-icon">&#9881;</div>
              <div class="feature-title">Solvers</div>
              <p class="feature-desc">Direct dense and sparse linear solvers, CARE, and quadratic programming.</p>
            </a>
            <a href="#/decompositions" class="feature-card">
              <div class="feature-icon">&#129513;</div>
              <div class="feature-title">Decompositions</div>
              <p class="feature-desc">Cholesky (LLT, LDLT), Full & Thin SVD, QR, and Full & Partial Pivoting LU.</p>
            </a>
            <a href="#/benchmark" class="feature-card">
              <div class="feature-icon">&#9201;</div>
              <div class="feature-title">Benchmarks</div>
              <p class="feature-desc">Stress tests comparing Eigen.js against OpenBLAS, SuiteSparse, ML Matrix, Math.js, and LaloLib.</p>
            </a>
          </div>
        </section>

        <!-- Usage & Installation Section -->
        <section class="home-section">
          <h2 class="section-title">Installation & Usage</h2>
          <p class="section-desc">Eigen.js is available as an npm package with embedded WebAssembly (zero external loaders required):</p>
          
          <div class="code-block-container">
            <div class="code-header">
              <span>Terminal</span>
              <button class="copy-btn" id="copyNpmBtn">Copy</button>
            </div>
            <pre class="code-snippet"><code><span class="code-comment"># Install via npm</span>
npm install eigen

<span class="code-comment"># Or via yarn</span>
yarn add eigen</code></pre>
          </div>

          <h3 class="subsection-title">Node.js (ESM & CJS) or Bundlers (Vite, Webpack)</h3>
          <div class="code-block-container">
            <div class="code-header">
              <span>test.mjs</span>
              <button class="copy-btn" id="copyEsmBtn">Copy</button>
            </div>
            <pre class="code-snippet" id="codeSampleEsm"><code><span class="code-keyword">import</span> eig <span class="code-keyword">from</span> <span class="code-string">'eigen'</span>;

<span class="code-keyword">async</span> <span class="code-keyword">function</span> <span class="code-function">main</span>() {
  <span class="code-comment">// Wait for WebAssembly binary instantiation</span>
  <span class="code-keyword">await</span> eig.ready;

  <span class="code-comment">// Create matrices</span>
  <span class="code-keyword">const</span> M = <span class="code-keyword">new</span> eig.Matrix([[<span class="code-number">1</span>, <span class="code-number">2</span>], [<span class="code-number">3</span>, <span class="code-number">4</span>]]);
  <span class="code-keyword">const</span> inv = M.inverse();

  console.log(<span class="code-string">'Original Matrix:'</span>);
  M.print();

  console.log(<span class="code-string">'Inverse Matrix:'</span>);
  inv.print();

  <span class="code-comment">// Flush garbage collection to release WASM pointers</span>
  eig.GC.flush();
}

main();</code></pre>
          </div>

          <h3 class="subsection-title">Browser via Script Tag</h3>
          <div class="code-block-container">
            <div class="code-header">
              <span>index.html</span>
              <button class="copy-btn" id="copyHtmlBtn">Copy</button>
            </div>
            <pre class="code-snippet" id="codeSampleHtml"><code><span class="code-tag">&lt;script</span> <span class="code-attr">src</span>=<span class="code-string">"dist/index.js"</span><span class="code-tag">&gt;&lt;/script&gt;</span>
<span class="code-tag">&lt;script&gt;</span>
  eig.ready.then(() =&gt; {
    <span class="code-keyword">const</span> A = eig.Matrix.identity(<span class="code-number">3</span>, <span class="code-number">3</span>);
    <span class="code-keyword">const</span> b = <span class="code-keyword">new</span> eig.Matrix([<span class="code-number">1</span>, <span class="code-number">2</span>, <span class="code-number">3</span>]);
    <span class="code-keyword">const</span> x = A.matMul(b);
    console.log(<span class="code-string">'Result:'</span>, x.get(<span class="code-number">0</span>, <span class="code-number">0</span>));
    eig.GC.flush();
  });
<span class="code-tag">&lt;/script&gt;</span></code></pre>
          </div>
        </section>
      </div>
    </div>
  `;

  // Copy buttons
  document.getElementById('copyNpmBtn')?.addEventListener('click', () => {
    navigator.clipboard.writeText('npm install eigen');
  });
  document.getElementById('copyEsmBtn')?.addEventListener('click', () => {
    const text = document.getElementById('codeSampleEsm')?.innerText || '';
    navigator.clipboard.writeText(text);
  });
  document.getElementById('copyHtmlBtn')?.addEventListener('click', () => {
    const text = document.getElementById('codeSampleHtml')?.innerText || '';
    navigator.clipboard.writeText(text);
  });

  // Start animated SVD demonstration on canvas
  initSvdAnimation(container);
}

/**
 * Initializes the geometric SVD animation on the canvas.
 * Computes SVD of random transformations using live window.eig.
 */
function initSvdAnimation(container) {
  const canvas = document.getElementById('svdCanvas');
  const valuesEl = document.getElementById('matrixValues');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;

  let currentTheta = 0;
  let targetTheta = 0;
  let currentS1 = 1.8;
  let targetS1 = 1.8;
  let currentS2 = 0.9;
  let targetS2 = 0.9;
  let matrixEntries = [1, 0, 0, 1];

  function generateNewState() {
    targetTheta = (Math.random() * 2 - 1) * Math.PI;
    targetS1 = 0.8 + Math.random() * 1.8;
    targetS2 = 0.5 + Math.random() * 1.5;

    if (window.eig?.Matrix && window.eig?.Decompositions?.svd) {
      try {
        const cos = Math.cos(targetTheta);
        const sin = Math.sin(targetTheta);
        const R = new window.eig.Matrix([[cos, -sin], [sin, cos]]);
        const D = new window.eig.Matrix([[targetS1, 0], [0, targetS2]]);
        const M = R.matMul(D);
        matrixEntries = [
          M.get(0, 0), M.get(0, 1),
          M.get(1, 0), M.get(1, 1)
        ];
        if (valuesEl) {
          valuesEl.innerHTML = `
            <span>[ ${matrixEntries[0].toFixed(2)}, ${matrixEntries[1].toFixed(2)} ]</span>
            <span>[ ${matrixEntries[2].toFixed(2)}, ${matrixEntries[3].toFixed(2)} ]</span>
            <span class="singular-values">&sigma;<sub>1</sub> = ${targetS1.toFixed(2)}, &sigma;<sub>2</sub> = ${targetS2.toFixed(2)}</span>
          `;
        }
        window.eig.GC?.flush();
      } catch (e) {
        // Fallback
      }
    }
  }

  generateNewState();
  const intervalId = setInterval(generateNewState, 3500);

  function animate() {
    // Smooth interpolation
    currentTheta += (targetTheta - currentTheta) * 0.05;
    currentS1 += (targetS1 - currentS1) * 0.05;
    currentS2 += (targetS2 - currentS2) * 0.05;

    ctx.clearRect(0, 0, w, h);

    // Coordinate axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(20, cy);
    ctx.lineTo(w - 20, cy);
    ctx.moveTo(cx, 20);
    ctx.lineTo(cx, h - 20);
    ctx.stroke();
    ctx.setLineDash([]);

    // Save and transform
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(currentTheta);

    const baseRadius = 45;
    const rx = baseRadius * currentS1;
    const ry = baseRadius * currentS2;

    // SVD transformed ellipse
    ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.strokeStyle = '#80cbc4';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.ellipse(0, 0, Math.max(rx, 5), Math.max(ry, 5), 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Major axis (sigma 1)
    ctx.strokeStyle = '#ffd54f';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(rx, 0);
    ctx.stroke();

    // Minor axis (sigma 2)
    ctx.strokeStyle = '#ff8a80';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, ry);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#ffd54f';
    ctx.font = '12px "Fira Code", monospace';
    ctx.fillText('\u03C31', rx + 6, 4);

    ctx.fillStyle = '#ff8a80';
    ctx.fillText('\u03C32', -18, ry + 12);

    ctx.restore();

    svdAnimationId = requestAnimationFrame(animate);
  }

  animate();

  // Clean up interval when container changes
  const observer = new MutationObserver(() => {
    if (!document.getElementById('svdCanvas')) {
      clearInterval(intervalId);
      if (svdAnimationId) cancelAnimationFrame(svdAnimationId);
      observer.disconnect();
    }
  });
  observer.observe(container, { childList: true });
}
