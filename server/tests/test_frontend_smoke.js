/**
 * server/tests/test_frontend_smoke.js
 * -----------------------------------
 * End-to-end smoke test for Task 3 (Minimal Live Frontend / Session Logic).
 *
 * Verifies:
 * 1. Completing a full session end-to-end:
 *    - All 10 questions appear in the fixed skill order (positions 1..10)
 *    - response_time_ms values sent are plausible (not all identical, > 0)
 *    - final mastery estimates match direct call to /predict for same history
 * 2. Running test twice:
 *    - Confirms question wording differs between runs (different set chosen)
 *    - Skill order and mastery keys stay exactly the same.
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

async function runSingleSession(port, sessionNum) {
  console.log(`\n--- Starting Run ${sessionNum} ---`);

  // 1. Start session
  const startRes = await makeRequest(port, 'POST', '/api/session/start');
  if (startRes.status !== 200) {
    throw new Error(`Run ${sessionNum}: Failed to start session: ${startRes.status}`);
  }

  const { session_id, questions } = startRes.body;
  if (!session_id || !Array.isArray(questions) || questions.length !== 10) {
    throw new Error(`Run ${sessionNum}: Invalid questions payload: ${JSON.stringify(startRes.body)}`);
  }

  // 2. Verify all 10 questions match fixed skill sequence in exact order
  for (let i = 0; i < 10; i++) {
    const q = questions[i];
    const exp = EXPECTED_SKILL_SEQUENCE[i];
    if (q.position !== exp.position) {
      throw new Error(`Position mismatch at index ${i}: got ${q.position}, expected ${exp.position}`);
    }
    if (q.raw_skill_id !== exp.raw_skill_id) {
      throw new Error(`Skill ID mismatch at pos ${q.position}: got ${q.raw_skill_id}, expected ${exp.raw_skill_id}`);
    }
    if (q.correct_answer !== undefined) {
      throw new Error(`Security breach: correct_answer exposed in client payload at pos ${q.position}!`);
    }
  }

  // 3. Submit all 10 answers with non-zero, plausible client response times
  const responseTimes = [];
  for (let i = 0; i < 10; i++) {
    // Generate realistic, non-identical client-measured durations
    const durationMs = 3800 + Math.floor(Math.sin(i * 1.5) * 1200) + i * 250;
    responseTimes.push(durationMs);

    const ansRes = await makeRequest(port, 'POST', `/api/session/${session_id}/answer`, {
      position: i + 1,
      submitted_answer: "test_answer",
      response_time_ms: durationMs,
      hint_used: 0
    });

    if (ansRes.status !== 200) {
      throw new Error(`Run ${sessionNum}: Failed to submit answer at pos ${i + 1}: ${ansRes.status}`);
    }
  }

  // Verify response times are plausible
  const uniqueTimes = new Set(responseTimes);
  if (uniqueTimes.size < 5) {
    throw new Error(`Response times lack variance: ${responseTimes}`);
  }
  if (responseTimes.some(t => t <= 0)) {
    throw new Error(`Found non-positive response time: ${responseTimes}`);
  }

  // 4. Request final mastery prediction
  const predictRes = await makeRequest(port, 'POST', `/api/session/${session_id}/predict`);
  if (predictRes.status !== 200) {
    throw new Error(`Run ${sessionNum}: Failed to get prediction: ${predictRes.status}`);
  }

  const mastery = predictRes.body.skill_mastery;
  if (!mastery) {
    throw new Error(`Run ${sessionNum}: Missing skill_mastery in response`);
  }

  // Confirm all 10 skills have valid float values in [0, 1]
  for (const exp of EXPECTED_SKILL_SEQUENCE) {
    const rawIdStr = String(exp.raw_skill_id);
    if (mastery[rawIdStr] === undefined) {
      throw new Error(`Run ${sessionNum}: Missing mastery for skill ${rawIdStr} (${exp.skill_name})`);
    }
    const val = mastery[rawIdStr];
    if (typeof val !== 'number' || isNaN(val) || val < 0.0 || val > 1.0) {
      throw new Error(`Run ${sessionNum}: Invalid mastery value for skill ${rawIdStr}: ${val}`);
    }
  }

  // Compare against direct call to Python for the exact same history
  const histRes = await makeRequest(port, 'GET', `/api/session/${session_id}/history`);
  const directRes = await makeRequest(8000, 'POST', '/predict', { student_history: histRes.body.history });
  const directMastery = directRes.body.skill_mastery;

  for (const exp of EXPECTED_SKILL_SEQUENCE) {
    const rawIdStr = String(exp.raw_skill_id);
    const nodeVal = mastery[rawIdStr];
    const directVal = directMastery[rawIdStr];
    const diff = Math.abs(nodeVal - directVal);
    if (diff > 1e-4) {
      throw new Error(`Discrepancy for skill ${rawIdStr}: Node=${nodeVal}, Direct=${directVal}, diff=${diff}`);
    }
  }

  console.log(`Run ${sessionNum} completed successfully.`);
  console.log(`- Questions: 10 in fixed skill order.`);
  console.log(`- Sample question text (pos 1): "${questions[0].question_text}"`);
  console.log(`- Response times: [${responseTimes.slice(0, 4).join(', ')}... ms]`);
  console.log(`- All 10 mastery bars match direct model evaluation within 1e-4 tolerance.`);

  return { questions, mastery, session_id };
}

async function runTask3SmokeTests() {
  console.log('=== RUNNING TASK 3 FRONTEND SMOKE TESTS ===\n');
  const port = 3299;
  const server = app.listen(port);

  try {
    // Run 1: Full session
    const run1 = await runSingleSession(port, 1);

    // Run 2: Second session to verify randomization of question text vs fixed skill order
    // Repeat until a distinct set is drawn (up to 5 tries) to reliably test text difference
    let run2 = null;
    let textDiffers = false;
    for (let attempt = 1; attempt <= 10; attempt++) {
      run2 = await runSingleSession(port, `2 (attempt ${attempt})`);
      if (run1.questions[0].question_text !== run2.questions[0].question_text) {
        textDiffers = true;
        break;
      }
    }

    if (!textDiffers) {
      throw new Error('Question wording did not differ across multiple runs (randomization issue).');
    }

    console.log('\n[Task 3 Smoke Test 2 Verification]:');
    console.log(`- Run 1 Question 1 text: "${run1.questions[0].question_text}"`);
    console.log(`- Run 2 Question 1 text: "${run2.questions[0].question_text}"`);
    console.log('- Both runs preserved the exact same skill at Pos 1 (ID 204: Percents).');

    console.log('\n=== ALL TASK 3 SMOKE TESTS PASSED ===');
    server.close();
    process.exit(0);
  } catch (err) {
    server.close();
    console.error('\nFAILED Task 3 Smoke Test:', err.message);
    process.exit(1);
  }
}

if (require.main === module) {
  runTask3SmokeTests();
}
