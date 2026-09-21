import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import esbuild from 'esbuild';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('=== Building Eigen-js JavaScript bundles ===');

const wasmPath = path.join(rootDir, 'build', 'eigen_gen.wasm');
const genJsPath = path.join(rootDir, 'build', 'eigen_gen.js');
const distDir = path.join(rootDir, 'dist');

if (!fs.existsSync(wasmPath) || !fs.existsSync(genJsPath)) {
  console.error('Error: build/eigen_gen.wasm or build/eigen_gen.js not found.');
  console.error('Please run "npm run build:wasm" first.');
  process.exit(1);
}

fs.mkdirSync(distDir, { recursive: true });

// Copy raw wasm to dist as well
fs.copyFileSync(wasmPath, path.join(distDir, 'eigen_gen.wasm'));

// Read wasm and generate embedded base64 module
const wasmBuffer = fs.readFileSync(wasmPath);
const wasmBase64 = wasmBuffer.toString('base64');

const virtualWasmModule = `
// Auto-generated embedded WASM binary
const base64 = ${JSON.stringify(wasmBase64)};
let wasmBinary;
if (typeof Buffer !== 'undefined') {
  wasmBinary = Buffer.from(base64, 'base64');
} else {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  wasmBinary = bytes.buffer;
}
export default wasmBinary;
`;

const tempWasmFile = path.join(rootDir, 'src', '_wasm_embedded.js');
fs.writeFileSync(tempWasmFile, virtualWasmModule);

const nodeShimPlugin = {
  name: 'node-shims',
  setup(build) {
    build.onResolve({ filter: /^(node:)?(fs|crypto|path)$/ }, args => {
      return { path: args.path, namespace: 'node-shim' };
    });
    build.onLoad({ filter: /.*/, namespace: 'node-shim' }, () => {
      return {
        contents: `
let fsMod = typeof globalThis.process !== 'undefined' && globalThis.process.versions?.node ? globalThis.require?.('node:fs') : {};
export default fsMod;
export const readFileSync = fsMod?.readFileSync;
export const readFile = fsMod?.readFile;
export const randomFillSync = () => {};
export const randomBytes = () => {};
`,
        loader: 'js',
      };
    });
  }
};

const wasmPlugin = {
  name: 'wasm-loader',
  setup(build) {
    build.onResolve({ filter: /\.wasm$/ }, () => {
      return { path: tempWasmFile };
    });
  },
};

try {
  // 1. Build ESM bundle (dist/index.mjs)
  console.log('Building dist/index.mjs (ESM)...');
  await esbuild.build({
    entryPoints: [path.join(rootDir, 'src', 'eigen.mjs')],
    bundle: true,
    format: 'esm',
    outfile: path.join(distDir, 'index.mjs'),
    plugins: [wasmPlugin, nodeShimPlugin],
    target: ['es2020', 'node18'],
    minify: false,
  });

  // 2. Build CJS / Browser UMD bundle (dist/index.js)
  console.log('Building dist/index.js (IIFE / UMD)...');
  await esbuild.build({
    entryPoints: [path.join(rootDir, 'src', 'eigen.mjs')],
    bundle: true,
    format: 'iife',
    globalName: '__eig_bundle',
    outfile: path.join(distDir, 'index.js'),
    plugins: [wasmPlugin, nodeShimPlugin],
    target: ['es2020', 'node18'],
    footer: {
      js: `
const __eig = typeof __eig_bundle !== 'undefined' ? (__eig_bundle.default || __eig_bundle) : null;
if (typeof module !== 'undefined' && module.exports) {
  module.exports = __eig;
  module.exports.default = __eig;
}
if (typeof window !== 'undefined') {
  window.eig = __eig;
}
if (typeof globalThis !== 'undefined') {
  globalThis.eig = __eig;
}
`
    },
    minify: false,
  });

  // 3. Also copy dist bundle to demo/dist so demo is 100% self-contained for static hosting (e.g., GitHub Pages)
  const demoDistDir = path.join(rootDir, 'demo', 'dist');
  fs.mkdirSync(demoDistDir, { recursive: true });
  fs.copyFileSync(path.join(distDir, 'index.js'), path.join(demoDistDir, 'index.js'));
  fs.copyFileSync(path.join(distDir, 'eigen_gen.wasm'), path.join(demoDistDir, 'eigen_gen.wasm'));

  console.log('=== JavaScript build completed successfully! ===');
  console.log('Generated:');
  console.log(`- ${path.join(distDir, 'index.mjs')} (${(fs.statSync(path.join(distDir, 'index.mjs')).size / 1024).toFixed(1)} KB)`);
  console.log(`- ${path.join(distDir, 'index.js')} (${(fs.statSync(path.join(distDir, 'index.js')).size / 1024).toFixed(1)} KB)`);
  console.log(`- ${path.join(distDir, 'eigen_gen.wasm')} (${(fs.statSync(path.join(distDir, 'eigen_gen.wasm')).size / 1024).toFixed(1)} KB)`);
  console.log(`- Copied to ${path.join(demoDistDir, 'index.js')} for standalone demo`);
} finally {
  // Clean up temporary embedded wasm file
  if (fs.existsSync(tempWasmFile)) {
    fs.unlinkSync(tempWasmFile);
  }
}
