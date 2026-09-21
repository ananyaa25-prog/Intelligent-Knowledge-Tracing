/**
 * server/src/index.js
 * -------------------
 * Node/Express backend proxy layer for the GatedKT inference and session pipeline.
 *
 * Exposes:
 * - Session endpoints: /api/session/start, /api/session/:sessionId/answer,
 *   /api/session/:sessionId/history, /api/session/:sessionId/predict
 * - Direct prediction endpoint: /api/predict
 * - Demo Teacher dashboard endpoint: /api/dashboard/matrix
 *
 * NOTE: Strictly a proxy and request validation layer with graceful error
 * handling (returns 502/503 when Python is down). Contains NO ML/tensor logic.
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_BASE_URL = (process.env.PYTHON_BASE_URL || process.env.INFERENCE_SERVICE_URL || 'http://127.0.0.1:8000').replace(/\/predict\/?$/, '');
const UPSTREAM_TIMEOUT_MS = parseInt(process.env.UPSTREAM_TIMEOUT_MS || '6000', 10);

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Load metadata
let skillNames = {};
const skillNamesPath = path.join(__dirname, '..', '..', 'data', 'processed', 'skill_names.json');
if (fs.existsSync(skillNamesPath)) {
  try {
    skillNames = JSON.parse(fs.readFileSync(skillNamesPath, 'utf8'));
  } catch (e) {
    console.warn('[NodeProxy] Warning: Could not parse skill_names.json');
  }
}

let questionBank = null;
const qbPath = path.join(__dirname, '..', '..', 'question_bank.json');
if (fs.existsSync(qbPath)) {
  try {
    questionBank = JSON.parse(fs.readFileSync(qbPath, 'utf8'));
  } catch (e) {
    console.warn('[NodeProxy] Warning: Could not parse question_bank.json');
  }
}

let sampleStudents = [];
const sampleStudentsPath = path.join(__dirname, '..', '..', 'data', 'processed', 'sample_test_students.json');
if (fs.existsSync(sampleStudentsPath)) {
  try {
    sampleStudents = JSON.parse(fs.readFileSync(sampleStudentsPath, 'utf8'));
  } catch (e) {
    console.warn('[NodeProxy] Warning: Could not parse sample_test_students.json');
  }
}

/**
 * Forward HTTP requests to the Python FastAPI service with timeout and error handling.
 */
async function forwardToPython(endpointPath, method = 'GET', body = null) {
  const url = `${PYTHON_BASE_URL}${endpointPath}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  try {
    const options = {
      method,
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal
    };
    if (body !== null) {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);
    clearTimeout(timeoutId);

    const data = await response.json().catch(() => ({}));
    return { status: response.status, data };
  } catch (err) {
    clearTimeout(timeoutId);
    const isTimeout = err.name === 'AbortError';
    return {
      status: 502,
      error: {
        error: 'Python inference service unavailable',
        upstream_url: url,
        details: isTimeout ? `Timeout after ${UPSTREAM_TIMEOUT_MS}ms` : (err.cause?.message || err.message)
      }
    };
  }
}

// -------------------------------------------------------------
// Validation Helpers
// -------------------------------------------------------------

function validateInteraction(record, index) {
  const prefix = index !== undefined ? `Interaction at index ${index}` : 'Interaction';

  if (!record || typeof record !== 'object') {
    return `${prefix} must be an object`;
  }
  if (record.skill_id === undefined || record.skill_id === null || record.skill_id === '') {
    return `${prefix} is missing required field 'skill_id'`;
  }
  if (record.correct !== 0 && record.correct !== 1) {
    return `${prefix} field 'correct' must be 0 or 1, got ${record.correct}`;
  }
  if (typeof record.response_time_ms !== 'number' || isNaN(record.response_time_ms) || record.response_time_ms < 0) {
    return `${prefix} field 'response_time_ms' must be a non-negative number, got ${record.response_time_ms}`;
  }
  if (!Number.isInteger(record.attempt_count) || record.attempt_count < 1) {
    return `${prefix} field 'attempt_count' must be an integer >= 1, got ${record.attempt_count}`;
  }
  if (record.hint_used !== 0 && record.hint_used !== 1) {
    return `${prefix} field 'hint_used' must be 0 or 1, got ${record.hint_used}`;
  }
  return null;
}

function validateHistoryPayload(body) {
  if (!body || typeof body !== 'object') {
    return 'Request body must be a JSON object';
  }
  if (!Array.isArray(body.student_history)) {
    return "Request body must contain an array 'student_history'";
  }
  if (body.student_history.length === 0) {
    return "'student_history' array cannot be empty";
  }
  for (let i = 0; i < body.student_history.length; i++) {
    const error = validateInteraction(body.student_history[i], i);
    if (error) return error;
  }
  return null;
}

function validateAnswerPayload(body) {
  if (!body || typeof body !== 'object') {
    return 'Request body must be a JSON object';
  }
  if (!Number.isInteger(body.position) || body.position < 1 || body.position > 10) {
    return "Field 'position' must be an integer between 1 and 10";
  }
  if (body.submitted_answer === undefined || body.submitted_answer === null) {
    return "Field 'submitted_answer' is required";
  }
  if (typeof body.response_time_ms !== 'number' || isNaN(body.response_time_ms) || body.response_time_ms < 0) {
    return "Field 'response_time_ms' must be a non-negative number";
  }
  return null;
}

// -------------------------------------------------------------
// Routes
// -------------------------------------------------------------

// Health check
app.get('/api/health', async (req, res) => {
  const upstream = await forwardToPython('/health', 'GET');
  res.json({
    status: 'ok',
    proxy_port: PORT,
    upstream_url: PYTHON_BASE_URL,
    upstream_status: upstream.status,
    upstream_data: upstream.data || upstream.error
  });
});

// Session Management (Task 0 / Task 2)
app.post('/api/session/start', async (req, res) => {
  const result = await forwardToPython('/session/start', 'POST', {});
  if (result.error) return res.status(result.status).json(result.error);
  return res.status(result.status).json(result.data);
});

app.post('/api/session/:sessionId/answer', async (req, res) => {
  const { sessionId } = req.params;
  if (!sessionId || sessionId.trim() === '') {
    return res.status(400).json({ error: 'sessionId parameter is required' });
  }

  // Pre-forwarding validation
  const validationError = validateAnswerPayload(req.body);
  if (validationError) {
    return res.status(400).json({ error: 'Invalid answer payload', details: validationError });
  }

  const result = await forwardToPython(`/session/${encodeURIComponent(sessionId)}/answer`, 'POST', req.body);
  if (result.error) return res.status(result.status).json(result.error);
  return res.status(result.status).json(result.data);
});

app.get('/api/session/:sessionId/history', async (req, res) => {
  const { sessionId } = req.params;
  const result = await forwardToPython(`/session/${encodeURIComponent(sessionId)}/history`, 'GET');
  if (result.error) return res.status(result.status).json(result.error);
  return res.status(result.status).json(result.data);
});

app.post('/api/session/:sessionId/predict', async (req, res) => {
  const { sessionId } = req.params;
  const result = await forwardToPython(`/session/${encodeURIComponent(sessionId)}/predict`, 'POST', {});
  if (result.error) return res.status(result.status).json(result.error);
  return res.status(result.status).json(result.data);
});

// Direct predict endpoint (with full payload validation)
app.post('/api/predict', async (req, res) => {
  const validationError = validateHistoryPayload(req.body);
  if (validationError) {
    return res.status(400).json({ error: 'Invalid request payload', details: validationError });
  }

  const result = await forwardToPython('/predict', 'POST', req.body);
  if (result.error) return res.status(result.status).json(result.error);
  return res.status(result.status).json(result.data);
});

// Backward compatibility for student history endpoint
app.post('/api/student/:studentId/predict', async (req, res) => {
  const validationError = validateHistoryPayload(req.body);
  if (validationError) {
    return res.status(400).json({ error: 'Invalid request payload', details: validationError });
  }
  const result = await forwardToPython('/predict', 'POST', req.body);
  if (result.error) return res.status(result.status).json(result.error);
  return res.status(result.status).json(result.data);
});

// Skills and question bank metadata
app.get('/api/skills', (req, res) => {
  res.json({
    skill_names: skillNames,
    skill_sequence: questionBank ? questionBank.skill_sequence : []
  });
});

app.get('/api/sample_students', (req, res) => {
  res.json({
    students: sampleStudents.map(s => ({
      student_id: s.student_id,
      num_interactions: s.num_interactions || s.history.length
    }))
  });
});

app.get('/api/sample_students/:studentId', (req, res) => {
  const found = sampleStudents.find(s => s.student_id.toLowerCase() === req.params.studentId.toLowerCase());
  if (!found) {
    return res.status(404).json({ error: `Sample student '${req.params.studentId}' not found` });
  }
  res.json(found);
});

// Part B: Multi-student dashboard endpoint (demo only, no research novelty)
app.get('/api/dashboard/matrix', async (req, res) => {
  if (sampleStudents.length === 0) {
    return res.status(500).json({ error: 'No sample students loaded' });
  }

  // 10 skills in fixed skill_sequence
  const fixedSkills = questionBank ? questionBank.skill_sequence : [
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

  // Select 12 real test-split students
  const subset = sampleStudents.slice(0, 12);
  const studentRows = [];

  for (const s of subset) {
    const upstreamResult = await forwardToPython('/predict', 'POST', { student_history: s.history });
    const mastery = (!upstreamResult.error && upstreamResult.status === 200)
      ? (upstreamResult.data.skill_mastery || {})
      : {};

    studentRows.push({
      student_id: s.student_id,
      num_interactions: s.num_interactions || s.history.length,
      mastery
    });
  }

  res.json({
    description: "a demo dashboard for visualizing model output across multiple students, built for presentation purposes — not a research contribution.",
    note: "Rows represent historical ASSISTments test-split students, NOT live-quiz participants.",
    skills: fixedSkills.map(s => ({
      skill_id: String(s.raw_skill_id),
      name: s.skill_name,
      position: s.position
    })),
    students: studentRows
  });
});

// Static frontend serving
app.use(express.static(path.join(__dirname, '..', 'public')));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'index.html'));
});

app.get('/dashboard', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'dashboard.html'));
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`[NodeProxy] Backend proxy listening on http://localhost:${PORT}`);
    console.log(`[NodeProxy] Forwarding to Python service at ${PYTHON_BASE_URL}`);
  });
}

module.exports = { app, validateInteraction, validateHistoryPayload, validateAnswerPayload };
