/**
 * server/tests/test_v2_routes.js
 * ------------------------------
 * Tests the adaptive-v2 pass-through routes of the Node proxy against a FAKE
 * upstream (a tiny local HTTP server standing in for FastAPI), so the test is
 * deterministic and needs neither Python nor the real database.
 *
 * Covers: GET /api/session/:id/profile, GET /api/student/:id/profile,
 * GET /api/class/overview (source validation), POST /api/session/:id/feedback
 * (payload validation), and that /api/session/:id/next passes new optional
 * fields (total_questions, decision) through unchanged.
 *
 * Run: node tests/test_v2_routes.js
 */

const assert = require('assert');
const http = require('http');

const seen = [];
const fake = http.createServer((req, res) => {
  let body = '';
  req.on('data', c => body += c);
  req.on('end', () => {
    seen.push({ method: req.method, url: req.url, body: body ? JSON.parse(body) : null });
    res.setHeader('Content-Type', 'application/json');
    if (req.url.startsWith('/session/abc/profile')) {
      return res.end(JSON.stringify({ session_id: 'abc', student_id: 'u1', profile_reliable: true, topics: [], unasked: [] }));
    }
    if (req.url.startsWith('/student/u1/profile')) {
      return res.end(JSON.stringify({ session_id: null, student_id: 'u1', profile_reliable: false, topics: [], unasked: [] }));
    }
    if (req.url.startsWith('/class/overview')) {
      return res.end(JSON.stringify({ students: [], status_counts: {}, needs_attention: [], source: req.url.split('=')[1] }));
    }
    if (req.url === '/session/abc/feedback') {
      return res.end(JSON.stringify({ ok: true }));
    }
    if (req.url === '/session/abc/next') {
      return res.end(JSON.stringify({ position: 3, total_questions: 17, policy: 'staircase',
        decision: { policy: 'staircase', reason: 'x' }, predicted_p: 0.5, is_complete: false }));
    }
    res.statusCode = 404;
    res.end(JSON.stringify({ detail: 'not found' }));
  });
});

function request(port, method, path, body = null) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : '';
    const req = http.request({ hostname: '127.0.0.1', port, path, method,
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) } }, (res) => {
      let out = '';
      res.on('data', c => out += c);
      res.on('end', () => resolve({ status: res.statusCode, body: out ? JSON.parse(out) : {} }));
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

async function main() {
  await new Promise(r => fake.listen(0, '127.0.0.1', r));
  const fakePort = fake.address().port;
  process.env.PYTHON_BASE_URL = `http://127.0.0.1:${fakePort}`;
  const { app } = require('../src/index');
  const server = app.listen(0);
  await new Promise(r => server.once('listening', r));
  const port = server.address().port;
  let passed = 0;
  const check = (name, fn) => { fn(); passed++; console.log(`✓ ${name}`); };

  try {
    let r = await request(port, 'GET', '/api/session/abc/profile');
    check('session profile proxied', () => { assert.strictEqual(r.status, 200); assert.strictEqual(r.body.session_id, 'abc'); });

    r = await request(port, 'GET', '/api/student/u1/profile');
    check('student profile proxied', () => { assert.strictEqual(r.status, 200); assert.strictEqual(r.body.student_id, 'u1'); });

    r = await request(port, 'GET', '/api/class/overview?source=sample');
    check('class overview proxied with source', () => { assert.strictEqual(r.status, 200); assert.strictEqual(r.body.source, 'sample'); });

    r = await request(port, 'GET', '/api/class/overview');
    check('class overview defaults to live', () => { assert.strictEqual(r.body.source, 'live'); });

    r = await request(port, 'GET', '/api/class/overview?source=bogus');
    check('class overview rejects bad source', () => assert.strictEqual(r.status, 400));

    r = await request(port, 'POST', '/api/session/abc/feedback', { perceived: 'about_right', helpful: 4 });
    check('feedback proxied', () => {
      assert.strictEqual(r.status, 200);
      const last = seen[seen.length - 1];
      assert.strictEqual(last.method, 'POST');
      assert.deepStrictEqual(last.body, { perceived: 'about_right', helpful: 4 });
    });

    r = await request(port, 'POST', '/api/session/abc/feedback', { perceived: 'meh', helpful: 9 });
    check('feedback validation', () => assert.strictEqual(r.status, 400));

    r = await request(port, 'POST', '/api/session/abc/next', {});
    check('/next passes new optional fields through', () => {
      assert.strictEqual(r.body.total_questions, 17);
      assert.strictEqual(r.body.decision.policy, 'staircase');
    });

    console.log(`\nALL ${passed} V2 ROUTE TESTS PASSED`);
    server.close(); fake.close();
    process.exit(0);
  } catch (err) {
    console.error('FAILED v2 route test:', err.message);
    server.close(); fake.close();
    process.exit(1);
  }
}

main();
