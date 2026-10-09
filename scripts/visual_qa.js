/**
 * scripts/visual_qa.js
 * --------------------
 * Browser QA for the adaptive-v2 UI (run via scripts/visual_qa.py, which starts
 * the real services on a temporary DB copy). Uses puppeteer-core from
 * .tools/qa/node_modules and an installed Chrome/Edge.
 *
 * Usage: node scripts/visual_qa.js <baseUrl> <outDir> <browserPath>
 * Writes screenshots + qa_result.json into outDir.
 */
const path = require('path');
const fs = require('fs');
const puppeteer = require(path.join(__dirname, '..', '.tools', 'qa', 'node_modules', 'puppeteer-core'));

const [BASE, OUT, BROWSER] = process.argv.slice(2);
fs.mkdirSync(OUT, { recursive: true });
const result = { base: BASE, viewports: {}, console_errors: [], screenshots: [], injected_states: [], notes: [] };
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function noOverflow(page) {
  return page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth,
    ok: document.documentElement.scrollWidth <= window.innerWidth }));
}

async function shot(page, name, full = false) {
  const file = path.join(OUT, name);
  await page.screenshot({ path: file, fullPage: full });
  result.screenshots.push(name);
}

async function playQuiz(page, tag, width, wantShots) {
  const vp = result.viewports[width];
  await page.goto(`${BASE}/`, { waitUntil: 'networkidle2' });
  vp.landing_overflow = await noOverflow(page);
  vp.context_cards = await page.$$eval('#contextCards .context-card', els => els.length).catch(() => 0);
  if (wantShots) await shot(page, `landing_${width}.png`, true);
  await page.type('#studentIdInput', `qa_${tag}_${Date.now() % 100000}`);
  await page.click('#loginStartBtn');
  const seenBands = new Set(), seenLevels = new Set();
  let questions = 0, verdictLeak = false;
  for (let i = 0; i < 25; i++) {
    const state = await Promise.race([
      page.waitForSelector('#submitBtn:not([disabled])', { visible: true, timeout: 30000 }).then(() => 'q'),
      page.waitForFunction(() => document.getElementById('resultsSection').style.display === 'block', { timeout: 30000 }).then(() => 'done')
    ]).catch(() => 'timeout');
    if (state !== 'q') break;
    questions++;
    const info = await page.evaluate(() => {
      const card = document.getElementById('predictionCard');
      const band = (card.className.match(/band-(\w+)/) || [])[1];
      const pill = document.getElementById('difficultyPill').className.match(/level-(\w+)/)[1];
      const text = document.getElementById('quizSection').innerText.toLowerCase();
      return { band, pill, cardShown: card.style.display !== 'none', reason: document.getElementById('decisionReason').textContent,
        counter: document.getElementById('questionProgressBadge').textContent,
        leak: /\b(correct!|incorrect|wrong answer|you got it)\b/.test(text) };
    });
    if (info.leak) verdictLeak = true;
    if (wantShots && !seenBands.has(info.band)) { seenBands.add(info.band); await shot(page, `quiz_band_${info.band}_${width}.png`); }
    if (wantShots && !seenLevels.has(info.pill)) { seenLevels.add(info.pill); await shot(page, `quiz_level_${info.pill}_${width}.png`); }
    seenBands.add(info.band); seenLevels.add(info.pill);
    vp.last_counter = info.counter;
    vp.reason_seen = vp.reason_seen || !!info.reason;
    await page.type('#answerInput', '1');
    await page.keyboard.press('Enter');   // Enter submits
    await sleep(150);
  }
  vp.questions_answered = questions;
  vp.bands_seen = [...seenBands];
  vp.levels_seen = [...seenLevels];
  vp.verdict_leak = verdictLeak;
  vp.quiz_overflow = await noOverflow(page);
  await page.waitForSelector('#resultsV2 .stat-row', { timeout: 60000 }).catch(() => null);
  await sleep(1500);
  vp.results_sections = await page.evaluate(() => ({
    stat_cards: document.querySelectorAll('#resultsV2 .stat-card').length,
    strong_box: !!document.querySelector('#resultsV2 .box-good'),
    weak_box: !!document.querySelector('#resultsV2 .box-bad'),
    topic_cards: document.querySelectorAll('#resultsV2 .topic-card').length,
    difficulty_profile: document.querySelectorAll('#resultsV2 .dp-topic').length,
    measured_rows: document.querySelectorAll('#resultsV2 .mvp-row').length,
    hatched_rows: document.querySelectorAll('#resultsV2 .mvp-fill.hatched').length,
    prediction_chart: !!document.getElementById('predVsActualChart'),
    feedback: !!document.getElementById('feedbackBox'),
    legacy_mastery_bars: document.querySelectorAll('#masteryBarsList .mastery-item').length,
    graph_canvas: !!document.querySelector('#skillGraphVis canvas')
  }));
  vp.results_overflow = await noOverflow(page);
  if (wantShots) await shot(page, `results_${width}.png`, true);
  // feedback round trip
  const fb = await page.evaluate(async () => {
    const b = document.querySelector('.fb-btn[data-perceived="about_right"]');
    if (!b) return 'missing';
    b.click(); document.querySelector('.fb-btn[data-helpful="4"]').click();
    document.getElementById('feedbackSend').click();
    await new Promise(r => setTimeout(r, 1200));
    return document.getElementById('feedbackMsg').textContent;
  });
  vp.feedback_message = fb;
  return { seenBands, seenLevels };
}

async function injectMissing(page, seenBands, seenLevels, width) {
  // Bands that never occurred in the real run are previewed by re-rendering the
  // CURRENT real question with a substituted predicted_p. Recorded as injected.
  const want = { good: 0.82, mid: 0.55, bad: 0.2 };
  const levels = ['easy', 'medium', 'hard'];
  await page.goto(`${BASE}/`, { waitUntil: 'networkidle2' });
  await page.type('#studentIdInput', `qa_inject_${Date.now() % 100000}`);
  await page.click('#loginStartBtn');
  await page.waitForSelector('#submitBtn:not([disabled])', { visible: true });
  for (const [band, p] of Object.entries(want)) {
    if (seenBands.has(band)) continue;
    await page.evaluate(pp => { const q = Object.assign({}, window.GKT.state.currentQuestion, { predicted_p: pp, is_probe: false }); window.GKT.onQuestion(q); }, p);
    await sleep(700);
    await shot(page, `quiz_band_${band}_${width}_injected.png`);
    result.injected_states.push(`band ${band} at ${width}px (predicted_p=${p} substituted on a real question)`);
  }
  for (const lv of levels) {
    if (seenLevels.has(lv)) continue;
    await page.evaluate(l => { const q = Object.assign({}, window.GKT.state.currentQuestion, { level: l }); window.GKT.onQuestion(q); }, lv);
    await sleep(500);
    await shot(page, `quiz_level_${lv}_${width}_injected.png`);
    result.injected_states.push(`level ${lv} pill at ${width}px (level substituted on a real question)`);
  }
}

async function dashboard(page, width) {
  const vp = result.viewports[width];
  await page.goto(`${BASE}/dashboard.html`, { waitUntil: 'networkidle2' });
  await sleep(2500);
  vp.dashboard_overflow = await noOverflow(page);
  vp.dashboard = await page.evaluate(() => ({
    overview_section: !!document.getElementById('classOverviewCard'),
    heatmap_rows: document.querySelectorAll('#classOverviewHeatmap tbody tr').length,
    attention_items: document.querySelectorAll('#needsAttentionList li').length,
    empty_state: (document.getElementById('classOverviewEmpty') || {}).textContent || ''
  }));
  await shot(page, `dashboard_${width}.png`, true);
  const simBtn = await page.$('#overviewSimBtn');
  if (simBtn) {
    await simBtn.click();
    await sleep(2500);
    vp.dashboard_sim = await page.evaluate(() => ({
      label: !!document.querySelector('#classOverviewCard .sim-label'),
      heatmap_rows: document.querySelectorAll('#classOverviewHeatmap tbody tr').length,
      empty_state: (document.getElementById('classOverviewEmpty') || {}).textContent || ''
    }));
    await shot(page, `dashboard_simulated_${width}.png`, true);
  }
}

(async () => {
  const browser = await puppeteer.launch({ executablePath: BROWSER, headless: 'new', args: ['--no-sandbox'] });
  try {
    for (const width of [1280, 375]) {
      result.viewports[width] = {};
      const page = await browser.newPage();
      await page.setViewport({ width, height: width === 375 ? 780 : 900, deviceScaleFactor: 1 });
      page.on('console', m => { if (m.type() === 'error') result.console_errors.push({ width, text: m.text() }); });
      page.on('pageerror', e => result.console_errors.push({ width, text: String(e) }));
      const seen = await playQuiz(page, `w${width}`, width, true);
      await injectMissing(page, seen.seenBands, seen.seenLevels, width);
      await dashboard(page, width);
      await page.close();
    }
  } catch (e) {
    result.notes.push(`QA run error: ${e.stack || e}`);
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(OUT, 'qa_result.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  }
})();
