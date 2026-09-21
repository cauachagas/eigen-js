import { chromium } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

const artifactDir = process.env.ARTIFACT_DIR || rootDir;

console.log('=== Running Playwright Browser Benchmark Test ===');

const browser = await chromium.launch({
  headless: true
});

const page = await browser.newPage({
  viewport: { width: 1200, height: 1000 }
});

page.on('console', msg => console.log('PAGE LOG:', msg.text()));
page.on('pageerror', err => console.log('PAGE ERROR:', err.message));

try {
  console.log('Navigating to http://localhost:3000...');
  await page.goto('http://localhost:3000', { waitUntil: 'domcontentloaded' });

  // 1. Check WASM status badge
  await page.waitForSelector('.status-badge.ready', { timeout: 10000 });
  const statusText = await page.locator('.status-badge .status-text').textContent();
  console.log('Status badge:', statusText);

  // 2. Test switching tabs under Matrix Multiplication
  const tabs = page.locator('#card_mat_mul .tab-btn');
  const count = await tabs.count();
  console.log(`Found ${count} library tabs for Matrix Multiplication:`);
  for (let i = 0; i < count; i++) {
    const tabName = await tabs.nth(i).textContent();
    await tabs.nth(i).click();
    await page.waitForTimeout(50);
    const code = await page.locator('#code_mat_mul').textContent();
    console.log(` - Tab [${tabName.trim()}]: active, code length = ${code.length}`);
  }

  // Switch back to Eigen JS
  await tabs.first().click();

  // 3. Click RUN ALL on Matrix Multiplication
  console.log('\nRunning RUN ALL on Matrix Multiplication...');
  await page.click('#runAll_mat_mul');

  // Wait for button to re-enable (finish)
  await page.waitForFunction(() => {
    const btn = document.querySelector('#runAll_mat_mul');
    return btn && !btn.disabled && !btn.textContent.includes('Running');
  }, { timeout: 30000 });

  // Collect results
  const results = await page.evaluate(() => {
    const cells = Array.from(document.querySelectorAll('#card_mat_mul .results-table td'));
    const headers = Array.from(document.querySelectorAll('#card_mat_mul .results-table th'));
    return headers.map((h, i) => ({
      library: h.textContent.trim(),
      result: cells[i]?.textContent.trim()
    }));
  });

  console.log('\n--- Matrix Multiplication Benchmark Results ---');
  results.forEach(r => console.log(`  ${r.library}: ${r.result}`));

  // 4. Click RUN ALL on Matrix Inversion
  console.log('\nRunning RUN ALL on Matrix Inversion...');
  await page.click('#runAll_mat_inv');
  await page.waitForFunction(() => {
    const btn = document.querySelector('#runAll_mat_inv');
    return btn && !btn.disabled && !btn.textContent.includes('Running');
  }, { timeout: 30000 });

  const invResults = await page.evaluate(() => {
    const cells = Array.from(document.querySelectorAll('#card_mat_inv .results-table td'));
    const headers = Array.from(document.querySelectorAll('#card_mat_inv .results-table th'));
    return headers.map((h, i) => ({
      library: h.textContent.trim(),
      result: cells[i]?.textContent.trim()
    }));
  });

  console.log('\n--- Matrix Inversion Benchmark Results ---');
  invResults.forEach(r => console.log(`  ${r.library}: ${r.result}`));

  // 5. Click RUN ALL on SVD
  console.log('\nRunning RUN ALL on Singular Value Decomposition...');
  await page.click('#runAll_mat_svd');
  await page.waitForFunction(() => {
    const btn = document.querySelector('#runAll_mat_svd');
    return btn && !btn.disabled && !btn.textContent.includes('Running');
  }, { timeout: 30000 });

  const svdResults = await page.evaluate(() => {
    const cells = Array.from(document.querySelectorAll('#card_mat_svd .results-table td'));
    const headers = Array.from(document.querySelectorAll('#card_mat_svd .results-table th'));
    return headers.map((h, i) => ({
      library: h.textContent.trim(),
      result: cells[i]?.textContent.trim()
    }));
  });

  console.log('\n--- SVD Benchmark Results ---');
  svdResults.forEach(r => console.log(`  ${r.library}: ${r.result}`));

  // 6. Click RUN ALL on Sparse Linear System (A * x = b)
  console.log('\nRunning RUN ALL on Sparse Linear System (A * x = b)...');
  await page.click('#runAll_sparse_solve');
  await page.waitForFunction(() => {
    const btn = document.querySelector('#runAll_sparse_solve');
    return btn && !btn.disabled && !btn.textContent.includes('Running');
  }, { timeout: 30000 });

  const sparseResults = await page.evaluate(() => {
    const cells = Array.from(document.querySelectorAll('#card_sparse_solve .results-table td'));
    const headers = Array.from(document.querySelectorAll('#card_sparse_solve .results-table th'));
    return headers.map((h, i) => ({
      library: h.textContent.trim(),
      result: cells[i]?.textContent.trim()
    }));
  });

  console.log('\n--- Sparse Linear System Benchmark Results ---');
  sparseResults.forEach(r => console.log(`  ${r.library}: ${r.result}`));

  // 7. Capture screenshot
  const screenshotPath = path.join(artifactDir, 'benchmark_screenshot.png');
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(`\nScreenshot saved to: ${screenshotPath}`);

  console.log('\n=== Playwright browser test succeeded! ===');
} catch (err) {
  console.error('Playwright test failed:', err);
} finally {
  await browser.close();
}
