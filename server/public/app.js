/**
 * server/public/app.js
 * --------------------
 * Frontend controller for Task 3: Live Quiz and Knowledge Tracing Tracker.
 *
 * Implements:
 * - Session initialization on load via POST /api/session/start
 * - True client-side response time measurement (performance.now())
 * - Step-by-step submission to POST /api/session/:sessionId/answer
 * - Final mastery estimation fetch via POST /api/session/:sessionId/predict
 * - Rendering of 10 per-skill mastery bars in fixed skill_sequence order
 */

let currentSessionId = null;
let questions = [];
let currentQuestionIndex = 0;
let questionStartTime = 0;
let timerInterval = null;

// DOM Elements
const quizSection = document.getElementById('quizSection');
const resultsSection = document.getElementById('resultsSection');
const progressBar = document.getElementById('progressBar');
const questionProgressBadge = document.getElementById('questionProgressBadge');
const questionSkillTag = document.getElementById('questionSkillTag');
const questionText = document.getElementById('questionText');
const answerForm = document.getElementById('answerForm');
const answerInput = document.getElementById('answerInput');
const submitBtn = document.getElementById('submitBtn');
const liveTimerText = document.getElementById('liveTimerText');
const masteryBarsList = document.getElementById('masteryBarsList');
const restartQuizBtn = document.getElementById('restartQuizBtn');
const statusBanner = document.getElementById('statusBanner');

window.addEventListener('DOMContentLoaded', () => {
  startNewQuiz();
  if (restartQuizBtn) {
    restartQuizBtn.addEventListener('click', startNewQuiz);
  }
  if (answerForm) {
    answerForm.addEventListener('submit', handleAnswerSubmit);
  }
});

function showStatus(msg, isError = false) {
  if (!statusBanner) return;
  statusBanner.textContent = msg;
  statusBanner.className = isError ? 'status-alert status-error' : 'status-alert status-success';
  statusBanner.style.display = 'block';
  setTimeout(() => {
    statusBanner.style.display = 'none';
  }, 5000);
}

async function startNewQuiz() {
  if (timerInterval) clearInterval(timerInterval);
  quizSection.style.display = 'block';
  resultsSection.style.display = 'none';
  questionText.textContent = 'Starting assessment session...';
  submitBtn.disabled = true;

  try {
    const res = await fetch('/api/session/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Failed to start session' }));
      throw new Error(err.error || `Server responded with ${res.status}`);
    }

    const data = await res.json();
    currentSessionId = data.session_id;
    questions = data.questions || [];
    currentQuestionIndex = 0;

    if (questions.length !== 10) {
      throw new Error(`Expected 10 questions, received ${questions.length}`);
    }

    renderCurrentQuestion();
  } catch (err) {
    console.error('Quiz start error:', err);
    showStatus(`Failed to start quiz: ${err.message}`, true);
    questionText.textContent = 'Unable to load quiz session. Please ensure the backend is running.';
  }
}

function renderCurrentQuestion() {
  const q = questions[currentQuestionIndex];
  const pos = currentQuestionIndex + 1;

  // Update UI metadata
  questionProgressBadge.textContent = `Question ${pos} of ${questions.length}`;
  progressBar.style.width = `${(pos / questions.length) * 100}%`;
  questionSkillTag.textContent = `Skill: ${q.skill_name} (ID: ${q.raw_skill_id})`;
  questionText.textContent = q.question_text;

  // Reset answer input
  answerInput.value = '';
  answerInput.disabled = false;
  submitBtn.disabled = false;
  answerInput.focus();

  // Begin client-measured response time tracking
  questionStartTime = performance.now();
  if (timerInterval) clearInterval(timerInterval);
  liveTimerText.textContent = 'Response time: 0.0s';
  timerInterval = setInterval(() => {
    const elapsedSec = ((performance.now() - questionStartTime) / 1000).toFixed(1);
    liveTimerText.textContent = `Response time: ${elapsedSec}s`;
  }, 100);
}

async function handleAnswerSubmit(e) {
  e.preventDefault();
  const submittedAnswer = answerInput.value.trim();
  if (!submittedAnswer) return;

  // Measure response time client-side
  if (timerInterval) clearInterval(timerInterval);
  const responseTimeMs = Math.round(performance.now() - questionStartTime);

  const q = questions[currentQuestionIndex];
  submitBtn.disabled = true;
  answerInput.disabled = true;

  try {
    const res = await fetch(`/api/session/${encodeURIComponent(currentSessionId)}/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        position: q.position,
        submitted_answer: submittedAnswer,
        response_time_ms: responseTimeMs,
        hint_used: 0
      })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Error submitting answer' }));
      throw new Error(err.error || `Server responded with ${res.status}`);
    }

    // Move to next question or complete quiz
    currentQuestionIndex++;
    if (currentQuestionIndex < questions.length) {
      renderCurrentQuestion();
    } else {
      await finishQuiz();
    }
  } catch (err) {
    console.error('Answer submit error:', err);
    showStatus(`Submission error: ${err.message}`, true);
    submitBtn.disabled = false;
    answerInput.disabled = false;
  }
}

async function finishQuiz() {
  if (timerInterval) clearInterval(timerInterval);
  quizSection.style.display = 'none';
  resultsSection.style.display = 'block';
  masteryBarsList.innerHTML = '<p style="color: var(--text-muted);">Estimating per-skill mastery via GatedKT...</p>';

  try {
    const res = await fetch(`/api/session/${encodeURIComponent(currentSessionId)}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Failed to compute mastery' }));
      throw new Error(err.error || `Server responded with ${res.status}`);
    }

    const data = await res.json();
    const mastery = data.skill_mastery || {};
    renderMasteryBars(mastery);
  } catch (err) {
    console.error('Mastery prediction error:', err);
    showStatus(`Prediction error: ${err.message}`, true);
    masteryBarsList.innerHTML = '<p style="color: var(--accent-rose);">Failed to calculate mastery predictions.</p>';
  }
}

function renderMasteryBars(mastery) {
  masteryBarsList.innerHTML = '';

  // Render bars for all 10 skills in exact fixed skill_sequence order
  questions.forEach(q => {
    const rawIdStr = String(q.raw_skill_id);
    const score = mastery[rawIdStr] !== undefined ? mastery[rawIdStr] : 0.5;
    const pct = Math.round(score * 100);

    let barColor = '#10b981'; // Green (high mastery)
    if (pct < 50) {
      barColor = '#f43f5e'; // Rose/Red (low mastery)
    } else if (pct < 75) {
      barColor = '#f59e0b'; // Amber (moderate mastery)
    }

    const item = document.createElement('div');
    item.className = 'mastery-item';
    item.innerHTML = `
      <div class="mastery-meta">
        <span>${q.position}. ${q.skill_name} <span style="color: var(--text-muted); font-size: 0.8rem;">(ID ${q.raw_skill_id})</span></span>
        <span style="color: ${barColor}; font-weight: 700;">${pct}%</span>
      </div>
      <div class="mastery-bar-bg">
        <div class="mastery-bar-fill" style="width: 0%; background: ${barColor};"></div>
      </div>
    `;

    masteryBarsList.appendChild(item);

    // Trigger smooth animation
    setTimeout(() => {
      const fill = item.querySelector('.mastery-bar-fill');
      if (fill) fill.style.width = `${pct}%`;
    }, 50);
  });
}
