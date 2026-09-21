/**
 * Documentation View - Live interactive API reference for Matrix, Solvers & Decompositions
 */

import { docsData } from '../data/docs.js';

function extractText(node) {
  if (!node) return '';
  if (typeof node === 'string') return node;
  if (node.value) return node.value;
  if (node.children && Array.isArray(node.children)) {
    return node.children.map(extractText).join(node.type === 'paragraph' ? ' ' : '');
  }
  return '';
}

function formatResult(val) {
  if (val === null || val === undefined) return String(val);

  if (typeof val === 'number') {
    return Number.isInteger(val) ? String(val) : val.toFixed(4);
  }
  if (typeof val === 'string' || typeof val === 'boolean') {
    return String(val);
  }

  // Eigen Matrix
  if (val && typeof val.rows === 'function' && typeof val.cols === 'function') {
    const rows = val.rows();
    const cols = val.cols();
    const isVector = cols === 1 || rows === 1;
    let lines = [`Matrix ${rows} \u00D7 ${cols}:`];
    for (let i = 0; i < Math.min(rows, 10); i++) {
      let row = [];
      for (let j = 0; j < Math.min(cols, 10); j++) {
        const v = val.get(i, j);
        row.push(typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : String(v));
      }
      lines.push('  [ ' + row.join(', ') + (cols > 10 ? ', ...' : '') + ' ]');
    }
    if (rows > 10) lines.push('  ... (' + (rows - 10) + ' more rows)');
    return lines.join('\n');
  }

  // Array
  if (Array.isArray(val)) {
    return val.map((item, idx) => `[${idx}]:\n${formatResult(item)}`).join('\n\n');
  }

  // Object (e.g. SVDResult, LUResult, QRResult)
  if (typeof val === 'object') {
    let out = [];
    for (let key in val) {
      if (Object.prototype.hasOwnProperty.call(val, key)) {
        const item = val[key];
        if (item && typeof item.rows === 'function') {
          out.push(`${key}:\n${formatResult(item)}`);
        } else {
          out.push(`${key}: ${formatResult(item)}`);
        }
      }
    }
    return out.join('\n\n');
  }

  return String(val);
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightJs(code) {
  const escaped = escapeHtml(code);
  return escaped.replace(
    /(\/\/[^\n]*)|('[\s\S]*?'|"[\s\S]*?")|(\b(?:const|let|var|for|function|return|new|async|await)\b)|(\b(?:Matrix|Solvers|Decompositions|ComplexDenseMatrix|SparseMatrix|TripletVector|SimplicialCholesky|SparseLU|eig)\b)|(\b(?:identity|ones|constant|random|diagonal|fromArray|get|set|rows|cols|length|dot|norm|rank|det|sum|block|mul|div|matAdd|matSub|matMul|hcat|vcat|inverse|transpose|cholesky|lu|qr|svd|eigenSolve|careSolve|quadProgSolve|solve)\b)|(\b\d+(?:\.\d+)?\b)/g,
    (match, comment, str, kw, cls, fn, num) => {
      if (comment) return `<span class="code-comment">${comment}</span>`;
      if (str) return `<span class="code-string">${str}</span>`;
      if (kw) return `<span class="code-keyword">${kw}</span>`;
      if (cls) return `<span class="code-class">${cls}</span>`;
      if (fn) return `<span class="code-function">${fn}</span>`;
      if (num) return `<span class="code-number">${num}</span>`;
      return match;
    }
  );
}

const exampleCodeStore = new Map();

export function renderDocsView(container, className) {
  const classData = docsData.find(c => c.name.toLowerCase() === className.toLowerCase());

  if (!classData) {
    container.innerHTML = `
      <div class="docs-wrapper">
        <div class="docs-not-found">
          <h2>Class not found</h2>
          <p>The requested class documentation for "${className}" does not exist.</p>
        </div>
      </div>
    `;
    return;
  }

  const name = classData.name;
  const description = extractText(classData.description);
  const constructorFunction = classData.constructorComment;
  const staticMembers = classData.members?.static || [];
  const instanceMembers = classData.members?.instance || [];

  container.innerHTML = `
    <div class="docs-wrapper">
      <!-- Header -->
      <div class="docs-header">
        <div class="docs-title-row">
          <h1 class="docs-class-name">${name}</h1>
          <span class="badge badge-class">Class</span>
        </div>
        <p class="docs-class-desc">${description || 'Documentation for ' + name}</p>
      </div>

      <!-- Content -->
      <div class="docs-content">
        ${constructorFunction ? renderMethodCard(constructorFunction, 'constructor', name) : ''}

        ${staticMembers.length > 0 ? `
          <div class="methods-group">
            <h2 class="methods-group-title">Static Methods</h2>
            <div class="methods-list">
              ${staticMembers.map(m => renderMethodCard(m, 'static', name)).join('')}
            </div>
          </div>
        ` : ''}

        ${instanceMembers.length > 0 ? `
          <div class="methods-group">
            <h2 class="methods-group-title">Instance Methods</h2>
            <div class="methods-list">
              ${instanceMembers.map(m => renderMethodCard(m, 'instance', name)).join('')}
            </div>
          </div>
        ` : ''}
      </div>
    </div>
  `;

  // Attach live code execution listeners
  attachExampleRunners(container);
}

function renderMethodCard(method, scope, parentClass) {
  const isConstructor = scope === 'constructor';
  const methodName = isConstructor ? `new ${parentClass}` : (method.name || method.namespace);
  const desc = extractText(method.description);
  const params = method.params || [];
  const returns = method.returns || [];
  const warnings = (method.tags || []).filter(t => t.title === 'warning');
  const examples = method.examples || [];
  const cardId = `method_${scope}_${(method.name || parentClass).replace(/[^a-zA-Z0-9]/g, '_')}_${Math.floor(Math.random()*100000)}`;

  return `
    <div class="method-card" id="${cardId}">
      <div class="method-header">
        <div class="method-title-wrap">
          <span class="method-name">${methodName}</span>
          <span class="scope-chip chip-${scope}">${scope}</span>
        </div>
      </div>

      ${desc ? `<p class="method-desc">${desc}</p>` : ''}

      ${warnings.map(w => `
        <div class="warning-box">
          <span class="warning-icon">&#9888;</span>
          <span class="warning-text">${extractText(w.description) || w.description}</span>
        </div>
      `).join('')}

      <!-- Parameters Table -->
      ${params.length > 0 ? `
        <div class="params-section">
          <h4 class="subhead">Parameters</h4>
          <table class="params-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              ${params.map(p => `
                <tr>
                  <td class="param-name"><code>${p.name}</code></td>
                  <td class="param-type"><code>${p.type?.name || 'any'}</code></td>
                  <td class="param-desc">${extractText(p.description) || '-'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      ` : ''}

      <!-- Returns Table -->
      ${returns.length > 0 ? `
        <div class="returns-section">
          <h4 class="subhead">Returns</h4>
          <div class="returns-row">
            <span class="return-type"><code>${returns[0].type?.name || 'void'}</code></span>
            <span class="return-desc">${extractText(returns[0].description) || ''}</span>
          </div>
        </div>
      ` : ''}

      <!-- Interactive Examples -->
      ${examples.length > 0 ? `
        <div class="examples-section">
          <h4 class="subhead">Interactive Example</h4>
          ${examples.map((ex, idx) => {
            const rawCode = ex.description;
            const exId = `${cardId}_ex_${idx}`;
            exampleCodeStore.set(exId, rawCode);
            return `
              <div class="example-box" id="${exId}">
                <div class="example-code-wrapper">
                  <pre class="example-pre"><code>${highlightJs(rawCode)}</code></pre>
                  <button class="run-example-btn" data-ex-id="${exId}" title="Run in live WebAssembly engine" aria-label="Run example">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                      <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    <span>Run</span>
                  </button>
                </div>
                <div class="example-output" id="${exId}_output" style="display: none;">
                  <div class="output-header">
                    <span class="output-title">Result:</span>
                    <button class="output-close-btn" data-close-id="${exId}_output">&times;</button>
                  </div>
                  <pre class="output-pre" id="${exId}_result"></pre>
                </div>
                <div class="example-error" id="${exId}_error" style="display: none;"></div>
              </div>
            `;
          }).join('')}
        </div>
      ` : ''}
    </div>
  `;
}

function attachExampleRunners(container) {
  // Run button handlers
  const runButtons = container.querySelectorAll('.run-example-btn');
  runButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      const exId = btn.getAttribute('data-ex-id');
      const exBox = document.getElementById(exId);
      if (!exBox) return;

      const codePre = exBox.querySelector('.example-pre code');
      const outputDiv = document.getElementById(`${exId}_output`);
      const resultPre = document.getElementById(`${exId}_result`);
      const errorDiv = document.getElementById(`${exId}_error`);

      try {
        if (window.eig?.ready) {
          await window.eig.ready;
        }
      } catch (e) {
        if (errorDiv) {
          errorDiv.textContent = 'Eigen.js failed to initialize: ' + e.message;
          errorDiv.style.display = 'block';
        }
        return;
      }

      if (!window.eig?.Matrix) {
        if (errorDiv) {
          errorDiv.textContent = 'Eigen.js WebAssembly runtime is still initializing. Please wait a second and try again.';
          errorDiv.style.display = 'block';
        }
        return;
      }

      const rawCode = exampleCodeStore.get(exId) || codePre?.innerText || '';

      try {
        if (errorDiv) errorDiv.style.display = 'none';

        // Wrap code in a function providing eig
        let evalCode = rawCode;
        if (!evalCode.includes('return ') && !evalCode.includes('const ') && !evalCode.includes('let ')) {
          evalCode = 'return ' + evalCode;
        }

        const fn = new Function('eig', evalCode);
        const result = fn(window.eig);

        if (resultPre) {
          resultPre.textContent = formatResult(result);
        }
        if (outputDiv) {
          outputDiv.style.display = 'block';
        }

        // Clean up allocated memory
        window.eig?.GC?.flush();
      } catch (err) {
        if (errorDiv) {
          errorDiv.textContent = 'Execution error: ' + err.message;
          errorDiv.style.display = 'block';
        }
        if (outputDiv) {
          outputDiv.style.display = 'none';
        }
      }
    });
  });

  // Close output handlers
  const closeButtons = container.querySelectorAll('.output-close-btn');
  closeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const outputId = btn.getAttribute('data-close-id');
      const outputDiv = document.getElementById(outputId);
      if (outputDiv) outputDiv.style.display = 'none';
    });
  });
}
