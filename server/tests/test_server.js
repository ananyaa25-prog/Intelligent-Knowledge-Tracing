/**
 * server/tests/test_server.js
 * ----------------------------
 * Smoke test suite for Task 2 (Node/Express backend layer).
 *
 * Verifies:
 * 1. Full round trip: start a session via Node, submit all 10 answers via Node,
 *    request mastery prediction via Node. Assert matches direct Python call.
 * 2. Downstream downtime: when Python service is unreachable, Node returns a clear
 *    502/503 error, does not hang, does not crash.
 * 3. Malformed requests: Node validation layer rejects bad payloads with 4xx
 *    BEFORE forwarding upstream.
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

async function runTests() {
  console.log('=== RUNNING TASK 2 (NODE/EXPRESS) SMOKE TESTS ===\n');
  const serverPort = 3199;
  const server = app.listen(serverPort);

  try {
    // -------------------------------------------------------------
    // Smoke Test 3: Malformed requests rejected with 4xx BEFORE forwarding
    // -------------------------------------------------------------
    console.log('[Test 3] Testing malformed payloads (Node validation layer)...');

    // 3a. Missing student_history on /api/predict
    const res3a = await makeRequest(serverPort, 'POST', '/api/predict', {});
    if (res3a.status !== 400 || !res3a.body.details.includes('student_history')) {
      throw new Error(`Expected 400 for missing student_history, got ${res3a.status}: ${JSON.stringify(res3a.body)}`);
    }

    // 3b. Missing position on submit-answer
    const res3b = await makeRequest(serverPort, 'POST', '/api/session/sess_test/answer', {
      submitted_answer: "20",
      response_time_ms: 4500
    });
    if (res3b.status !== 400 || !res3b.body.details.includes('position')) {
      throw new Error(`Expected 400 for missing position, got ${res3b.status}: ${JSON.stringify(res3b.body)}`);
    }

    // 3c. Invalid position (> 10)
    const res3c = await makeRequest(serverPort, 'POST', '/api/session/sess_test/answer', {
      position: 15,
      submitted_answer: "20",
      response_time_ms: 4500
    });
    if (res3c.status !== 400 || !res3c.body.details.includes('position')) {
      throw new Error(`Expected 400 for position > 10, got ${res3c.status}: ${JSON.stringify(res3c.body)}`);
    }

    // 3d. Negative response_time_ms
    const res3d = await makeRequest(serverPort, 'POST', '/api/session/sess_test/answer', {
      position: 1,
      submitted_answer: "20",
      response_time_ms: -100
    });
    if (res3d.status !== 400 || !res3d.body.details.includes('response_time_ms')) {
      throw new Error(`Expected 400 for negative response_time_ms, got ${res3d.status}: ${JSON.stringify(res3d.body)}`);
    }

    // 3e. Invalid correct value in interaction
    const res3e = await makeRequest(serverPort, 'POST', '/api/predict', {
      student_history: [{ skill_id: 204, correct: 5, response_time_ms: 1000, attempt_count: 1, hint_used: 0 }]
    });
    if (res3e.status !== 400 || !res3e.body.details.includes('correct')) {
      throw new Error(`Expected 400 for invalid correct value, got ${res3e.status}: ${JSON.stringify(res3e.body)}`);
    }

    console.log('[Test 3 PASS] All malformed payloads rejected with 400 by Node validation layer.\n');

    // -------------------------------------------------------------
    // Smoke Test 2: Downstream service downtime handling
    // -------------------------------------------------------------
    console.log('[Test 2] Testing upstream downtime handling (Python service unreachable)...');

    // Create a temporary express app pointed to a dead port
    process.env.PYTHON_BASE_URL = 'http://127.0.0.1:59999';
    // Dynamically delete cache and reload index with dead port
    delete require.cache[require.resolve('../src/index')];
    const { app: deadApp } = require('../src/index');
    const deadServer = deadApp.listen(3198);

    const t0 = Date.now();
    const res2 = await makeRequest(3198, 'POST', '/api/session/start', {});
    const elapsed = Date.now() - t0;

    deadServer.close();

    if (res2.status !== 502) {
      throw new Error(`Expected 502 for dead upstream, got ${res2.status}: ${JSON.stringify(res2.body)}`);
    }
    if (!res2.body.error || !res2.body.error.toLowerCase().includes('unavailable')) {
      throw new Error(`Expected clear error message, got: ${JSON.stringify(res2.body)}`);
    }
    console.log(`[Test 2 PASS] Downstream outage handled gracefully with status 502 in ${elapsed}ms (no hang, no crash).\n`);

    // -------------------------------------------------------------
    // Smoke Test 1: Full round trip through Node
    // -------------------------------------------------------------
    console.log('[Test 1] Testing full round trip through Node vs direct Python call...');

    // Restore standard upstream port (8000)
    process.env.PYTHON_BASE_URL = 'http://127.0.0.1:8000';
    delete require.cache[require.resolve('../src/index')];
    const { app: liveApp } = require('../src/index');
    const liveServer = liveApp.listen(3197);

    // Check if Python service is actually running on port 8000
    const healthCheck = await makeRequest(3197, 'GET', '/api/health');
    if (healthCheck.status !== 200 || healthCheck.body.upstream_status !== 200) {
      console.log('  Notice: Python service on 8000 is not currently running.');
      console.log('  Starting session round-trip test will require running Python service.');
      liveServer.close();
      server.close();
      return { skippedRoundTrip: true };
    }

    // 1. Start session via Node
    const startRes = await makeRequest(3197, 'POST', '/api/session/start');
    if (startRes.status !== 200) {
      throw new Error(`Failed to start session via Node: ${startRes.status} ${JSON.stringify(startRes.body)}`);
    }
    const sessionId = startRes.body.session_id;
    const questions = startRes.body.questions;
    if (!sessionId || !Array.isArray(questions) || questions.length !== 10) {
      throw new Error(`Invalid session start response: ${JSON.stringify(startRes.body)}`);
    }

    // 2. Submit all 10 answers via Node
    for (let pos = 1; pos <= 10; pos++) {
      const ansRes = await makeRequest(3197, 'POST', `/api/session/${sessionId}/answer`, {
        position: pos,
        submitted_answer: "test_answer",
        response_time_ms: 5000 + pos * 200
      });
      if (ansRes.status !== 200) {
        throw new Error(`Failed to submit answer for pos ${pos}: ${ansRes.status} ${JSON.stringify(ansRes.body)}`);
      }
    }

    // 3. Request mastery prediction via Node
    const predictRes = await makeRequest(3197, 'POST', `/api/session/${sessionId}/predict`);
    if (predictRes.status !== 200) {
      throw new Error(`Failed to get mastery prediction: ${predictRes.status} ${JSON.stringify(predictRes.body)}`);
    }

    const nodeMastery = predictRes.body.skill_mastery;
    if (!nodeMastery || Object.keys(nodeMastery).length !== 10) {
      throw new Error(`Expected 10 skills in mastery, got: ${JSON.stringify(nodeMastery)}`);
    }

    // 4. Verify match against direct call to Python for the same history
    const histRes = await makeRequest(3197, 'GET', `/api/session/${sessionId}/history`);
    const directRes = await makeRequest(8000, 'POST', '/predict', { student_history: histRes.body.history });

    const directMastery = directRes.body.skill_mastery;
    for (const [skillId, val] of Object.entries(nodeMastery)) {
      const diff = Math.abs(val - directMastery[skillId]);
      if (diff > 1e-4) {
        throw new Error(`Discrepancy for skill ${skillId}: Node=${val}, Direct=${directMastery[skillId]}`);
      }
    }

    console.log('[Test 1 PASS] Full 10-question round trip matched direct Python evaluation within 1e-4 tolerance across all 10 skills.\n');

    liveServer.close();
    server.close();
    console.log('=== ALL TASK 2 SMOKE TESTS PASSED ===');
    process.exit(0);
  } catch (err) {
    server.close();
    console.error('\nFAILED Task 2 Smoke Test:', err.message);
    process.exit(1);
  }
}

if (require.main === module) {
  runTests();
}

module.exports = { runTests };
