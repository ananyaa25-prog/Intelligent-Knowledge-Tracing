/**
 * server/scripts/run_visual_qa.js
 * Comprehensive automated visual QA script using puppeteer-core and msedge.exe
 */

const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const ARTIFACT_DIR = 'C:\\Users\\Ananya Ambastha\\.gemini\\antigravity-ide\\brain\\94360063-c9ba-4fdd-9923-39f80b752049';

async function run() {
  console.log('=== STARTING PRESENTATION-READINESS VISUAL QA SWEEP ===\n');

  if (!fs.existsSync(ARTIFACT_DIR)) {
    fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
  }

  const browser = await puppeteer.launch({
    executablePath: EDGE_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1280,1024']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 1024 });

  const consoleErrors = [];
  const consoleWarnings = [];

  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push({ url: page.url(), text: msg.text() });
    } else if (msg.type() === 'warning') {
      consoleWarnings.push({ url: page.url(), text: msg.text() });
    }
  });

  page.on('pageerror', err => {
    consoleErrors.push({ url: page.url(), text: err.toString() });
  });

  // -------------------------------------------------------------
  // TEST 1: LIVE QUIZ PAGE & FULL FLOW
  // -------------------------------------------------------------
  console.log('[1/4] Navigating to Live Quiz at http://localhost:3000/ ...');
  await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });

  // Check login elements
  await page.waitForSelector('#studentIdInput');
  await page.type('#studentIdInput', 'demo_student_qa');
  await page.click('#loginStartBtn');

  console.log('  Submitting 10 questions sequentially...');
  for (let pos = 1; pos <= 10; pos++) {
    await page.waitForSelector('#submitBtn:not([disabled])', { visible: true });
    await page.waitForSelector('#answerInput:not([disabled])', { visible: true });
    await page.type('#answerInput', '42');
    await page.click('#submitBtn');
    // Wait for the button to become disabled or question to advance
    await new Promise(r => setTimeout(r, 400));
  }

  // Wait for results card
  console.log('  Waiting for Results Mastery Card...');
  await page.waitForSelector('#resultsSection[style*="display: block"]', { visible: true, timeout: 60000 });
  await page.waitForSelector('.mastery-item', { visible: true, timeout: 60000 });

  // Check Task 19 Synthesis Summary
  await page.waitForSelector('#masterySummaryCard', { visible: true });
  const summaryText = await page.$eval('#masterySummaryText', el => el.textContent);
  console.log('\n  ✓ Task 19 Auto-Generated Mastery Summary Captured:');
  console.log(`    "${summaryText}"\n`);

  // Assert no technical jargon in summary
  const forbiddenJargon = ['gate', 'ppmi', 'embedding', 'gru', 'gat', 'recurrent', 'tensor', 'logits'];
  for (const term of forbiddenJargon) {
    const regex = new RegExp(`\\b${term}\\b`, 'i');
    if (regex.test(summaryText)) {
      throw new Error(`Jargon violation: Found forbidden term "${term}" in student summary!`);
    }
  }
  console.log('  ✓ Verified ZERO technical jargon in summary.');

  // Check Task 18 Vis Network Tooltips Bug Fix
  console.log('  Testing Vis Network tooltips for <br> and raw HTML leaks...');
  await new Promise(r => setTimeout(r, 1500)); // Allow graph stabilization

  // Extract created tooltips directly via evaluating DOM helper functions
  const tooltipCheck = await page.evaluate(() => {
    const dummyNode = { name: 'Percents', raw_skill_id: 204, interaction_count: 1 };
    const nodeEl = createNodeTooltip(dummyNode, 25, true, true);
    const edgeEl = createEdgeTooltip(1.85, 'Percents', 'Percent Discount');

    return {
      nodeHtml: nodeEl.outerHTML,
      nodeText: nodeEl.innerText,
      hasLiteralBrInNodeText: nodeEl.innerText.includes('<br>'),
      edgeHtml: edgeEl.outerHTML,
      edgeText: edgeEl.innerText,
      hasLiteralBrInEdgeText: edgeEl.innerText.includes('<br>')
    };
  });

  if (tooltipCheck.hasLiteralBrInNodeText || tooltipCheck.hasLiteralBrInEdgeText) {
    throw new Error('Tooltip Bug: Literal <br> string found in tooltip text!');
  }
  console.log('  ✓ Verified tooltips render via DOM HTMLElement containers with NO literal <br> strings.');
  console.log(`    Node tooltip text sample:\n${tooltipCheck.nodeText.split('\n').map(l => '      ' + l).join('\n')}`);
  console.log(`    Edge tooltip text sample:\n${tooltipCheck.edgeText.split('\n').map(l => '      ' + l).join('\n')}`);

  // Take Screenshot 1: Quiz Results & Mastery Summary
  const screenshot1Path = path.join(ARTIFACT_DIR, 'quiz_mastery_results.png');
  await page.screenshot({ path: screenshot1Path, fullPage: false });
  console.log(`  ✓ Saved Screenshot 1: ${screenshot1Path}`);

  // Scroll to graph & trend and take Screenshot 2
  await page.evaluate(() => window.scrollBy(0, 500));
  await new Promise(r => setTimeout(r, 500));
  const screenshot2Path = path.join(ARTIFACT_DIR, 'skill_relation_graph_and_trend.png');
  await page.screenshot({ path: screenshot2Path, fullPage: false });
  console.log(`  ✓ Saved Screenshot 2: ${screenshot2Path}`);

  // -------------------------------------------------------------
  // TEST 2: TEACHER DASHBOARD
  // -------------------------------------------------------------
  console.log('\n[2/4] Navigating to Teacher Dashboard at http://localhost:3000/dashboard.html ...');
  await page.goto('http://localhost:3000/dashboard.html', { waitUntil: 'networkidle0' });

  await page.waitForSelector('#heatmapTable');
  await page.waitForSelector('.heatmap-table tr');

  // Check toggle buttons
  await page.click('#tabSample');
  await new Promise(r => setTimeout(r, 800));
  const sampleStudentCount = await page.$eval('#statStudentCount', el => el.textContent);
  console.log(`  ✓ Sample toggle works: loaded ${sampleStudentCount} historical students.`);

  await page.click('#tabLive');
  await new Promise(r => setTimeout(r, 800));
  const liveStudentCount = await page.$eval('#statStudentCount', el => el.textContent);
  console.log(`  ✓ Live toggle works: loaded ${liveStudentCount} registered live students.`);

  // Click a student row to inspect the Drilldown Modal
  const firstStudentRow = await page.$('tbody tr td.sticky-col');
  if (firstStudentRow) {
    await firstStudentRow.click();
    await page.waitForSelector('#studentModal', { visible: true });
    console.log('  ✓ Clicked student row: drilldown modal opened successfully.');
    await new Promise(r => setTimeout(r, 800));
    await page.click('#modalCloseBtn');
    await new Promise(r => setTimeout(r, 400));
  }

  // Take Screenshot 3: Teacher Dashboard
  const screenshot3Path = path.join(ARTIFACT_DIR, 'teacher_dashboard_heatmap.png');
  await page.screenshot({ path: screenshot3Path, fullPage: false });
  console.log(`  ✓ Saved Screenshot 3: ${screenshot3Path}`);

  // -------------------------------------------------------------
  // TEST 3: MODEL INTERNALS & GATE FUSION VIEW (TASK 20)
  // -------------------------------------------------------------
  console.log('\n[3/4] Navigating to Relocated Model Internals at http://localhost:3000/model-internals.html ...');
  await page.goto('http://localhost:3000/model-internals.html', { waitUntil: 'networkidle0' });

  // Verify framing disclaimer sentence
  const disclaimerText = await page.$eval('.disclaimer-banner', el => el.textContent);
  console.log(`  ✓ Framing disclaimer verified:\n    "${disclaimerText.trim()}"`);

  // Click "Load Latest Session"
  console.log('  Clicking "Load Latest Session"...');
  await page.click('#loadRecentBtn');
  await page.waitForSelector('#gateVisualizerCard', { visible: true, timeout: 10000 });
  await page.waitForSelector('#gatePlotImg', { visible: true, timeout: 10000 });

  // Take Screenshot 4: Model Internals
  const screenshot4Path = path.join(ARTIFACT_DIR, 'model_internals_gate_fusion.png');
  await page.screenshot({ path: screenshot4Path, fullPage: false });
  console.log(`  ✓ Saved Screenshot 4: ${screenshot4Path}`);

  // -------------------------------------------------------------
  // AUDIT: ZERO CONSOLE ERRORS ON NORMAL USER FLOW
  // -------------------------------------------------------------
  console.log('\n[Normal User Flow Console Audit]:');
  console.log(`  • Unhandled errors during normal flow:   ${consoleErrors.length}`);
  console.log(`  • Warnings encountered:                 ${consoleWarnings.length}`);

  if (consoleErrors.length > 0) {
    console.error('\n  [FAIL] Console Errors detected during normal flow:');
    consoleErrors.forEach((e, i) => console.error(`    ${i + 1}. [${e.url}] ${e.text}`));
    await browser.close();
    process.exit(1);
  }
  console.log('  ✓ ZERO browser console errors detected across normal user flows!');

  // -------------------------------------------------------------
  // TEST 4: ERROR STATE SIMULATION & VERIFICATION
  // -------------------------------------------------------------
  console.log('\n[4/4] Verifying Error States and Backend Failure Handling...');
  
  // Test invalid session error state on model-internals.html
  await page.type('#sessionIdInput', 'invalid-nonexistent-session-id');
  await page.click('#inspectBtn');
  await page.waitForSelector('#errorStateCard', { visible: true });
  const errorMsg = await page.$eval('#errorMessage', el => el.textContent);
  console.log(`  ✓ Error handling verified: human-readable error displayed ("${errorMsg.trim()}").`);

  // Take Screenshot 5: Error State Card
  const screenshot5Path = path.join(ARTIFACT_DIR, 'error_state_handling.png');
  await page.screenshot({ path: screenshot5Path, fullPage: false });
  console.log(`  ✓ Saved Screenshot 5 (Error State): ${screenshot5Path}`);

  console.log('\n=== ALL PRESENTATION-READINESS & SMOKE TESTS PASSED ===\n');

  await browser.close();
}

run().catch(err => {
  console.error('\n[FATAL ERROR during visual QA]:', err);
  process.exit(1);
});
