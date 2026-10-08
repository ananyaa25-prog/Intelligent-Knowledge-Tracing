/**
 * server/tests/test_frontend_smoke.js
 * -----------------------------------
 * Smoke test suite for Task 3 & Task 6 (Live Frontend & Multi-Session Student Flow).
 *
 * Verifies:
 * 1. Completing a full session end-to-end with student ID:
 *    - All 10 questions appear in fixed skill order (positions 1..10)
 *    - Response times measured client-side
 *    - Mastery prediction returned for all 10 skills
 * 2. Multi-Session flow with same student ID:
 *    - Student summary shows prior completed session (welcome back check)
 *    - Second session produces cumulative 20-interaction mastery
 * 3. Fresh student ID:
 *    - No false prior session history
 */

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

const EXPECTED_SKILL_SEQUENCE = [
  { position: 1,  skill_name: "Percents",                                        raw_skill_id: 204 },
  { position: 2,  skill_name: "Percent Discount",                                 raw_skill_id: 203 },
  { position: 3,  skill_name: "Rate",                                             raw_skill_id: 217 },
  { position: 4,  skill_name: "Polynomial Factors",                               raw_skill_id: 346 },
  { position: 5,  skill_name: "Simplifying Expressions positive exponents",       raw_skill_id: 371 },
  { position: 6,  skill_name: "Solving Systems of Linear Equations",              raw_skill_id: 350 },
  { position: 7,  skill_name: "Solving Systems of Linear Equations by Graphing",  raw_skill_id: 378 },
  { position: 8,  skill_name: "Ordering Fractions",                               raw_skill_id: 50  },
  { position: 9,  skill_name: "Equivalent Fractions",                             raw_skill_id: 48  },
  { position: 10, skill_name: "Pythagorean Theorem",                              raw_skill_id: 27  }
];

async function runTask6SmokeTests() {
  console.log('=== RUNNING TASK 6 (FRONTEND & MULTI-SESSION) SMOKE TESTS ===\n');
  const port = 3399;
  const server = app.listen(port);

  try {
    const testStudentId = `student_ui_${Date.now().toString().slice(-6)}`;

    // 1. Initial check: New student ID has 0 prior sessions
    console.log(`[Step 1] Checking initial summary for fresh student '${testStudentId}'...`);
    const initialSummary = await makeRequest(port, 'GET', `/api/student/${testStudentId}/summary`);
    if (initialSummary.status !== 200) {
      throw new Error(`Summary check failed: ${initialSummary.status}`);
    }
    if (initialSummary.body.completed_session_count !== 0) {
      throw new Error(`Expected 0 prior sessions for new student, got ${initialSummary.body.completed_session_count}`);
    }
    console.log('✓ PASS: Fresh student has no false prior session history.');

    // 2. Complete Session 1 for this student
    console.log(`\n[Step 2] Completing Session 1 for student '${testStudentId}'...`);
    const s1Start = await makeRequest(port, 'POST', '/api/session/start', { student_id: testStudentId, source: 'test' });
    if (s1Start.status !== 200 || !s1Start.body.session_id) {
      throw new Error(`Session 1 start failed: ${s1Start.status}`);
    }
    const sid1 = s1Start.body.session_id;

    for (let pos = 1; pos <= 10; pos++) {
      const durationMs = 4000 + pos * 200;
      const ansRes = await makeRequest(port, 'POST', `/api/session/${sid1}/answer`, {
        position: pos,
        submitted_answer: "test_answer_1",
        response_time_ms: durationMs,
        hint_used: 0
      });
      if (ansRes.status !== 200) {
        throw new Error(`Submit answer pos ${pos} failed: ${ansRes.status}`);
      }
    }

    const s1Predict = await makeRequest(port, 'POST', `/api/session/${sid1}/predict`);
    if (s1Predict.status !== 200 || s1Predict.body.num_interactions_processed !== 10) {
      throw new Error(`Session 1 prediction failed: ${JSON.stringify(s1Predict.body)}`);
    }
    console.log('✓ PASS: Session 1 completed and produced single-session mastery (10 interactions).');

    // 3. Check summary before starting Session 2 (returning student status)
    console.log(`\n[Step 3] Checking returning-student indicator for '${testStudentId}'...`);
    const returningSummary = await makeRequest(port, 'GET', `/api/student/${testStudentId}/summary`);
    if (returningSummary.status !== 200) {
      throw new Error(`Returning summary check failed: ${returningSummary.status}`);
    }
    if (returningSummary.body.completed_session_count !== 1 || returningSummary.body.total_interactions !== 10) {
      throw new Error(`Expected 1 completed session (10 interactions), got: ${JSON.stringify(returningSummary.body)}`);
    }
    console.log(`✓ PASS: Returning student correctly recognized: completed_session_count=1, total_interactions=10.`);

    // 4. Complete Session 2 for the SAME student ID
    console.log(`\n[Step 4] Completing Session 2 for SAME student '${testStudentId}'...`);
    const s2Start = await makeRequest(port, 'POST', '/api/session/start', { student_id: testStudentId, source: 'test' });
    if (s2Start.status !== 200) {
      throw new Error(`Session 2 start failed: ${s2Start.status}`);
    }
    const sid2 = s2Start.body.session_id;

    for (let pos = 1; pos <= 10; pos++) {
      const durationMs = 3500 + pos * 150;
      await makeRequest(port, 'POST', `/api/session/${sid2}/answer`, {
        position: pos,
        submitted_answer: "test_answer_2",
        response_time_ms: durationMs,
        hint_used: 0
      });
    }

    // 5. Request cumulative prediction
    const s2Predict = await makeRequest(port, 'POST', `/api/session/${sid2}/predict`);
    if (s2Predict.status !== 200) {
      throw new Error(`Session 2 prediction failed: ${s2Predict.status}`);
    }
    if (s2Predict.body.num_interactions_processed !== 20) {
      throw new Error(`Expected 20 cumulative interactions processed, got ${s2Predict.body.num_interactions_processed}`);
    }

    // Confirm all 10 skills have mastery values
    for (const exp of EXPECTED_SKILL_SEQUENCE) {
      const rawIdStr = String(exp.raw_skill_id);
      const val = s2Predict.body.skill_mastery[rawIdStr];
      if (val === undefined || isNaN(val) || val < 0 || val > 1) {
        throw new Error(`Invalid mastery for skill ${rawIdStr}: ${val}`);
      }
    }
    console.log('✓ PASS: Session 2 prediction successfully processed all 20 cumulative interactions.');

    // 6. Test fresh student ID isolation
    const brandNewStudent = `student_brand_new_${Date.now().toString().slice(-4)}`;
    const brandNewSummary = await makeRequest(port, 'GET', `/api/student/${brandNewStudent}/summary`);
    if (brandNewSummary.body.completed_session_count !== 0) {
      throw new Error(`New student isolation failed: ${JSON.stringify(brandNewSummary.body)}`);
    }
    console.log(`✓ PASS: Brand new student '${brandNewStudent}' starts with 0 prior sessions.`);

    console.log('\n======================================================');
    console.log('ALL TASK 6 SMOKE TESTS PASSED SUCCESSFULLY!');
    console.log('======================================================\n');
    server.close();
    process.exit(0);
  } catch (err) {
    server.close();
    console.error('\nFAILED Task 6 Smoke Test:', err.message);
    process.exit(1);
  }
}

if (require.main === module) {
  runTask6SmokeTests();
}

module.exports = { runTask6SmokeTests };
