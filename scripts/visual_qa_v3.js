/**
 * scripts/visual_qa_v3.js
 * -----------------------
 * v3 browser QA (run via scripts/visual_qa_v3.py, real services on a temp DB copy):
 * for viewport widths 1280 and 375 and for the sections given on the command line,
 * picks the learner level on the login card, plays a whole session through the UI,
 * and checks: picker present, section badge shown, multiple-choice buttons rendered
 * for mc items, no correct/incorrect verdict during the quiz, no horizontal overflow,
 * results banner with the section and its validation label. Also opens the dashboard
 * and checks the sections card. Screenshots + qa_v3_result.json go to outDir.
 *
 * Usage: node scripts/visual_qa_v3.js <baseUrl> <outDir> <browserPath> <section,section,...>
 */
const path = require('path');
const fs = require('fs');
const puppeteer = require(path.join(__dirname, '..', '.tools', 'qa', 'node_modules', 'puppeteer-core'));

const [BASE, OUT, BROWSER, SECTIONS] = process.argv.slice(2);
fs.mkdirSync(OUT, { recursive: true });
const result = { base: BASE, runs: [], dashboard: {}, console_errors: [], screenshots: [] };

async function overflow(page) {
  return page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
}
async function shot(page, name, full = false) {
  await page.screenshot({ path: path.join(OUT, name), fullPage: full });
  result.screenshots.push(name);
}

async function play(page, section, width) {
  const run = { section, width, questions: 0, mc_items: 0, mc_buttons_ok: true, verdict_leak: false, overflow_ok: true };
  await page.goto(`${BASE}/`, { waitUntil: 'networkidle2' });
  await page.waitForSelector('#sectionSelect', { timeout: 10000 }).catch(() => null);
  run.picker = !!(await page.$('#sectionSelect'));
  run.picker_options = await page.$$eval('#sectionSelect option', os => os.map(o => o.value)).catch(() => []);
  if (run.picker) await page.select('#sectionSelect', section);
  run.overflow_ok = run.overflow_ok && await overflow(page);
  await shot(page, `v3_login_${section || 'default'}_${width}.png`, true);
  await page.type('#studentIdInput', `qa_v3_${section || 'default'}_${width}_${Date.now() % 100000}`);
  await page.click('#loginStartBtn');
  let firstShot = true, mcShot = false;
  for (let i = 0; i < 25; i++) {
    const state = await Promise.race([
      page.waitForSelector('#submitBtn:not([disabled])', { visible: true, timeout: 30000 }).then(() => 'q'),
      page.waitForFunction(() => document.getElementById('resultsSection').style.display === 'block', { timeout: 30000 }).then(() => 'done')
    ]).catch(() => 'timeout');
    if (state !== 'q') { run.end_state = state; break; }
    run.questions++;
    const info = await page.evaluate(() => {
      const box = document.getElementById('answerChoices');
      const text = document.getElementById('quizSection').innerText.toLowerCase();
      return { badge: (document.getElementById('sectionBadge') || {}).textContent || '',
        choices: box && box.style.display !== 'none' ? box.querySelectorAll('.choice-btn').length : 0,
        question: document.getElementById('questionText').textContent,
        leak: /\b(correct!|incorrect|wrong answer|you got it)\b/.test(text) };
    });
    run.badge = run.badge || info.badge;
    if (info.leak) run.verdict_leak = true;
    const isMc = /\bA\) /.test(info.question);
    if (isMc) { run.mc_items++; if (info.choices < 2) run.mc_buttons_ok = false; }
    run.overflow_ok = run.overflow_ok && await overflow(page);
    if (firstShot) { await shot(page, `v3_quiz_${section || 'default'}_${width}.png`); firstShot = false; }
    if (isMc && info.choices >= 2) {
      if (!mcShot) { await shot(page, `v3_quiz_mc_${section || 'default'}_${width}.png`); mcShot = true; }
      await page.click('#answerChoices .choice-btn');   // fills the box only; still no verdict
    } else {
      await page.type('#answerInput', '1');
    }
    await page.click('#submitBtn');
  }
  await page.waitForFunction(() => document.getElementById('resultsSection').style.display === 'block', { timeout: 30000 }).catch(() => null);
  await new Promise(r => setTimeout(r, 2500));
  run.results_banner = await page.$eval('#sectionResultsBanner', e => e.textContent).catch(() => null);
  run.overflow_ok = run.overflow_ok && await overflow(page);
  await shot(page, `v3_results_${section || 'default'}_${width}.png`, true);
  result.runs.push(run);
}

(async () => {
  const browser = await puppeteer.launch({ executablePath: BROWSER, headless: 'new', args: ['--no-sandbox'] });
  try {
    for (const width of [1280, 375]) {
      const page = await browser.newPage();
      page.on('console', m => { if (m.type() === 'error') result.console_errors.push(m.text().slice(0, 200)); });
      page.on('pageerror', e => result.console_errors.push(String(e).slice(0, 200)));
      await page.setViewport({ width, height: width === 375 ? 780 : 900, deviceScaleFactor: 1 });
      for (const sec of SECTIONS.split(',')) await play(page, sec === 'default' ? '' : sec, width);
      await page.goto(`${BASE}/dashboard.html`, { waitUntil: 'networkidle2' });
      await page.waitForSelector('#sectionsInfoCard', { timeout: 10000 }).catch(() => null);
      result.dashboard[width] = {
        card: !!(await page.$('#sectionsInfoCard')),
        labels: await page.$$eval('#sectionsInfoCard .section-badge', bs => bs.map(b => b.textContent)).catch(() => []),
        overflow_ok: await overflow(page) };
      await shot(page, `v3_dashboard_${width}.png`, true);
      await page.close();
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync(path.join(OUT, 'qa_v3_result.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
})().catch(e => { console.error(e); process.exit(1); });
