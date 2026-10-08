/**
 * server/tests/test_dashboard_smoke.js
 * ------------------------------------
 * Smoke test suite for Part B & Task 7:
 * Live SQLite Teacher Dashboard & Historical Sample Toggle.
 *
 * Verifies:
 * 1. Honest Scope & Labeling: Confirms presentation disclaimer is present in payload.
 * 2. Historical Sample Data: /api/dashboard/matrix?source=sample returns ASSISTments sample.
 * 3. Live Data Multi-Student Query: Completes 3 real quiz sessions, confirms all 3 appear in live matrix.
 * 4. Multi-Session Live Refresh: 1 student completes a 2nd session, confirms row updates with 20 cumulative interactions on refresh.
 * 5. Spot-check cell values against direct FastAPI calls.
 */

const assert = require('assert');
const http = require('http');
const { app } = require('../src/index');

function makeRequest(port, method, path, body = null) {
  return new Promise((resolve, reject) => {
    const postData = body ? JSON.stringify(body) : '';
    const req = http.request({
      hostname: '127.0.0.1',
      port,
      path,
      method,
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
      }
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const parsed = data ? JSON.parse(data) : {};
          resolve({ status: res.statusCode, body: parsed });
        } catch (e) {
          resolve({ status: res.statusCode, raw: data });
        }
      });
    });

    req.on('error', reject);
    if (postData) req.write(postData);
    req.end();
  });
}

async function runDashboardSmokeTests() {
  console.log('=== RUNNING TASK 7 (LIVE TEACHER DASHBOARD) SMOKE TESTS ===\n');
  const port = 3499;
  const server = app.listen(port);

  try {
    // 1. Test Historical Sample View
    console.log('[Step 1] Fetching historical sample view (?source=sample)...');
    const sampleRes = await makeRequest(port, 'GET', '/api/dashboard/matrix?source=sample');
    assert.strictEqual(sampleRes.status, 200);
    const sampleData = sampleRes.body;

    assert(sampleData.description.includes('not a research contribution'), 'Missing honest disclaimer');
    assert(sampleData.note.includes('historical ASSISTments test-split'), 'Missing historical sample note');
    assert.strictEqual(sampleData.source, 'sample');
    assert(sampleData.students.length >= 10, 'Sample view should have >= 10 students');
    console.log(`✓ PASS: Sample view returned ${sampleData.students.length} historical students with honest labeling.`);

    // 2. Test Live View (SQLite Database)
    console.log('\n[Step 2] Fetching live database view (?source=live)...');
    const liveRes1 = await makeRequest(port, 'GET', '/api/dashboard/matrix?source=live');
    assert.strictEqual(liveRes1.status, 200);
    assert.strictEqual(liveRes1.body.source, 'live');
    assert(Array.isArray(liveRes1.body.students));
    console.log(`✓ PASS: Live database query succeeded (current live students: ${liveRes1.body.students.length}).`);

    // 3. Create 3 distinct real students and complete 1 session each
    const student1 = `dash_st_A_${Date.now().toString().slice(-4)}`;
    const student2 = `dash_st_B_${Date.now().toString().slice(-4)}`;
    const student3 = `dash_st_C_${Date.now().toString().slice(-4)}`;
    const testStudents = [student1, student2, student3];

    console.log(`\n[Step 3] Registering and completing 1 session for 3 real students:`, testStudents);
    for (const stId of testStudents) {
      const sStart = await makeRequest(port, 'POST', '/api/session/start', { student_id: stId, source: 'live' });
      const sid = sStart.body.session_id;
      for (let pos = 1; pos <= 10; pos++) {
        await makeRequest(port, 'POST', `/api/session/${sid}/answer`, {
          position: pos,
          submitted_answer: "test_ans",
          response_time_ms: 4000.0 + pos * 100,
          hint_used: 0
        });
      }
      await makeRequest(port, 'POST', `/api/session/${sid}/predict`);
    }
    console.log('✓ PASS: Completed 10 questions for all 3 students.');

    // 4. Query live dashboard matrix on refresh
    console.log('\n[Step 4] Refreshing live dashboard matrix to verify all 3 students appear...');
    const liveRes2 = await makeRequest(port, 'GET', '/api/dashboard/matrix?source=live');
    assert.strictEqual(liveRes2.status, 200);
    const liveStudents = liveRes2.body.students;

    for (const stId of testStudents) {
      const found = liveStudents.find(s => s.student_id === stId);
      assert(found, `Student ${stId} missing in live dashboard matrix`);
      assert.strictEqual(found.completed_sessions, 1);
      assert.strictEqual(found.num_interactions, 10);
      assert(Object.keys(found.mastery).length > 0, `No mastery for student ${stId}`);
    }
    console.log(`✓ PASS: All 3 new students appear with 1 completed session and 10 interactions each.`);

    // 5. Have Student 1 complete a SECOND session
    console.log(`\n[Step 5] Having student '${student1}' complete a SECOND session...`);
    const s2Start = await makeRequest(port, 'POST', '/api/session/start', { student_id: student1, source: 'live' });
    const sid2 = s2Start.body.session_id;
    for (let pos = 1; pos <= 10; pos++) {
      await makeRequest(port, 'POST', `/api/session/${sid2}/answer`, {
        position: pos,
        submitted_answer: "test_ans_2",
        response_time_ms: 5000.0,
        hint_used: 0
      });
    }
    await makeRequest(port, 'POST', `/api/session/${sid2}/predict`);

    // 6. Refresh live dashboard to confirm cumulative update
    console.log(`\n[Step 6] Refreshing live dashboard matrix to verify cumulative update for '${student1}'...`);
    const liveRes3 = await makeRequest(port, 'GET', '/api/dashboard/matrix?source=live');
    const updatedStudent1 = liveRes3.body.students.find(s => s.student_id === student1);
    assert(updatedStudent1, `Student ${student1} missing after second session`);
    assert.strictEqual(updatedStudent1.completed_sessions, 2, `Expected 2 completed sessions, got ${updatedStudent1.completed_sessions}`);
    assert.strictEqual(updatedStudent1.num_interactions, 20, `Expected 20 cumulative interactions, got ${updatedStudent1.num_interactions}`);
    console.log(`✓ PASS: Dashboard row for '${student1}' updated to 2 completed sessions (20 interactions) on live refresh.`);

    // 7. Spot-check mastery values against direct FastAPI call
    console.log(`\n[Step 7] Spot-checking student '${student1}' cumulative mastery against direct FastAPI call...`);
    const directRes = await makeRequest(8000, 'POST', `/student/${encodeURIComponent(student1)}/predict`);
    assert.strictEqual(directRes.status, 200);
    const directMastery = directRes.body.skill_mastery;

    for (const [skillId, val] of Object.entries(updatedStudent1.mastery)) {
      const directVal = directMastery[skillId];
      assert(directVal !== undefined, `Direct call missing skill ${skillId}`);
      const diff = Math.abs(val - directVal);
      assert(diff <= 1e-4, `Discrepancy for skill ${skillId}: dash=${val}, direct=${directVal}`);
    }
    console.log(`✓ PASS: Cumulative mastery matched direct FastAPI service identically.`);

    console.log('\n======================================================');
    console.log('ALL TASK 7 (LIVE DASHBOARD) SMOKE TESTS PASSED!');
    console.log('======================================================\n');
    server.close();
    process.exit(0);
  } catch (err) {
    server.close();
    console.error('\nFAILED Task 7 Smoke Test:', err);
    process.exit(1);
  }
}

if (require.main === module) {
  runDashboardSmokeTests();
}

module.exports = { runDashboardSmokeTests };
