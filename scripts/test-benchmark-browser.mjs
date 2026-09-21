import { chromium } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

const artifactDir = process.env.ARTIFACT_DIR || rootDir;

console.log('=== Running Playwright Full Site & Benchmark Test ===');

const browser = await chromium.launch({
  headless: true
});

const page = await browser.newPage({
  viewport: { width: 1280, height: 950 }
});

page.on('console', msg => {
  if (msg.type() === 'error') console.log('PAGE LOG [ERROR]:', msg.text());
});
page.on('pageerror', err => console.log('PAGE ERROR:', err.message));

try {
  // 1. Test Home View
  console.log('1. Navigating to Home view (http://localhost:3000/#/)...');
  await page.goto('http://localhost:3000/#/', { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.hero-title', { timeout: 10000 });
  const heroTitle = await page.locator('.hero-title').textContent();
  console.log(` - Hero title: "${heroTitle.trim()}"`);

  // Verify canvas exists
  const hasCanvas = await page.locator('#svdCanvas').isVisible();
  console.log(` - Live SVD Canvas visible: ${hasCanvas}`);

  // 2. Test Navigation Drawer & Matrix Documentation View
  console.log('\n2. Testing navigation to Matrix documentation (#/matrix)...');
  await page.click('#navMatrix');
  await page.waitForSelector('.docs-class-name', { timeout: 5000 });
  const className = await page.locator('.docs-class-name').textContent();
  console.log(` - Navigated to class: "${className.trim()}"`);

  // Verify methods rendered
  const methodsCount = await page.locator('.method-card').count();
  console.log(` - Rendered ${methodsCount} method cards for Matrix`);

  // Test running an interactive example
  console.log(' - Running interactive code example...');
  const runBtn = page.locator('.run-example-btn').first();
  await runBtn.click();
  const visibleOutput = page.locator('.example-output:not([style*="display: none"]) .output-pre').first();
  await visibleOutput.waitFor({ state: 'visible', timeout: 8000 });
  const exampleOutput = await visibleOutput.textContent();
  console.log(` - Example Output received:\n${exampleOutput.trim()}`);

  // 3. Test Navigation to Solvers Documentation View
  console.log('\n3. Testing navigation to Solvers documentation (#/solvers)...');
  await page.click('#navSolvers');
  await page.waitForFunction(() => document.querySelector('.docs-class-name')?.textContent === 'Solvers');
  console.log(' - Solvers documentation loaded successfully');

  // 4. Test Navigation to Decompositions Documentation View
  console.log('\n4. Testing navigation to Decompositions (#/decompositions)...');
  await page.click('#navDecompositions');
  await page.waitForFunction(() => document.querySelector('.docs-class-name')?.textContent === 'Decompositions');
  console.log(' - Decompositions documentation loaded successfully');

  // 5. Test Navigation to Benchmark View
  console.log('\n5. Testing navigation to Benchmark view (#/benchmark)...');
  await page.click('#navBenchmark');
  await page.waitForSelector('#card_sparse_solve', { timeout: 10000 });
  console.log(' - Benchmark view loaded with all cards');

  // Check WebAssembly status badge
  await page.waitForSelector('.status-badge.ready', { timeout: 10000 });
  const statusBadge = await page.locator('.status-badge .status-text').textContent();
  console.log(` - Status badge: "${statusBadge.trim()}"`);

  // 6. Run Sparse Linear System benchmark (Eigen SparseLU vs SuiteSparse UMFPACK)
  console.log('\n6. Running Sparse linear system benchmark...');
  const runAllSparseBtn = page.locator('#card_sparse_solve .run-all-btn');
  await runAllSparseBtn.click();
  await page.waitForFunction(() => {
    const btn = document.querySelector('#card_sparse_solve .run-all-btn');
    return btn && !btn.disabled && !btn.textContent.includes('RUNNING');
  }, { timeout: 30000 });

  const sparseResults = await page.evaluate(() => {
    const cells = Array.from(document.querySelectorAll('#card_sparse_solve .results-table td'));
    const headers = Array.from(document.querySelectorAll('#card_sparse_solve .results-table th'));
    return headers.map((h, i) => ({
      library: h.textContent.trim(),
      result: cells[i]?.textContent.trim()
    }));
  });

  console.log('--- Sparse Linear System Benchmark Results ---');
  sparseResults.forEach(r => console.log(`  ${r.library}: ${r.result}`));

  // 7. Test Hamburger toggle
  console.log('\n7. Testing Hamburger menu toggle...');
  const menuToggle = page.locator('#menuToggle');
  await menuToggle.click();
  const isCollapsed = await page.evaluate(() => document.getElementById('appLayout')?.classList.contains('drawer-collapsed'));
  console.log(` - Drawer collapsed: ${isCollapsed}`);
  await menuToggle.click(); // restore

  // Take screenshot
  const screenshotPath = path.join(artifactDir, 'benchmark_screenshot.png');
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(`\nScreenshot saved to: ${screenshotPath}`);

  console.log('\n=== Playwright test completed successfully! ===');
} catch (error) {
  console.error('Playwright Test Failed:', error);
  process.exit(1);
} finally {
  await browser.close();
}
