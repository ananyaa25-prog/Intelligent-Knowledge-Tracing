const puppeteer = require('puppeteer-core');
const path = require('path');
const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const ARTIFACT_DIR = 'C:\\Users\\Ananya Ambastha\\.gemini\\antigravity-ide\\brain\\94360063-c9ba-4fdd-9923-39f80b752049';

(async () => {
  const browser = await puppeteer.launch({ executablePath: EDGE_PATH, headless: 'new', args: ['--no-sandbox', '--window-size=1280,1024'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 1024 });
  await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });
  await page.type('#studentIdInput', 'demo_student_qa');
  await page.click('#loginStartBtn');
  for (let i = 1; i <= 10; i++) {
    await page.waitForSelector('#submitBtn:not([disabled])');
    await page.type('#answerInput', '42');
    await page.click('#submitBtn');
    await new Promise(r => setTimeout(r, 400));
  }
  await page.waitForSelector('#resultsSection[style*="display: block"]', { timeout: 60000 });
  await new Promise(r => setTimeout(r, 2000));
  
  const graphEl = await page.$('#graphVisualizerCard');
  if (graphEl) {
    await graphEl.scrollIntoView();
    await new Promise(r => setTimeout(r, 800));
    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'skill_relation_graph_view.png') });
  }

  const trendEl = await page.$('#trendVisualizerCard');
  if (trendEl) {
    await trendEl.scrollIntoView();
    await new Promise(r => setTimeout(r, 800));
    await page.screenshot({ path: path.join(ARTIFACT_DIR, 'mastery_trend_view.png') });
  }

  await browser.close();
  console.log('Graph and Trend screenshots captured successfully.');
})().catch(err => {
  console.error(err);
  process.exit(1);
});
