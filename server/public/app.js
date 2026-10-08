/**
 * server/public/app.js
 * --------------------
 * Frontend controller for Task 3 & Task 6:
 * Live Quiz and Cumulative Knowledge Tracing Mastery Tracker.
 *
 * Implements:
 * - Student ID entry screen with instant returning-student summary check
 * - Multi-session state persistence in SQLite
 * - Client-side response time measurement (performance.now())
 * - Step-by-step submission to POST /api/session/:sessionId/answer
 * - Cumulative mastery estimation fetch via POST /api/session/:sessionId/predict
 * - Rendering of 10 per-skill mastery bars labeled with cumulative interaction/session counts
 * - "Start New Test" flows (continue as same student vs new student ID)
 */

let currentStudentId = null;
let currentSessionId = null;
let currentQuestion = null;
let questionStartTime = 0;
let timerInterval = null;

// Fixed 10-topic skill sequence (mirrors question_bank.json). Adaptive sessions
// select the next topic at runtime, but the results screen still needs this
// fixed list to render per-skill mastery bars in stable order.
const SKILL_SEQUENCE = [
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

// DOM Elements
const loginSection = document.getElementById('loginSection');
const studentIdInput = document.getElementById('studentIdInput');
const loginForm = document.getElementById('loginForm');
const loginStartBtn = document.getElementById('loginStartBtn');
const studentWelcomeStatus = document.getElementById('studentWelcomeStatus');

const quizSection = document.getElementById('quizSection');
const resultsSection = document.getElementById('resultsSection');
const progressBar = document.getElementById('progressBar');
const activeStudentBadge = document.getElementById('activeStudentBadge');
const questionProgressBadge = document.getElementById('questionProgressBadge');
const questionSkillTag = document.getElementById('questionSkillTag');
const questionText = document.getElementById('questionText');
const predictionBanner = document.getElementById('predictionBanner');
const predictedPText = document.getElementById('predictedPText');
const answerForm = document.getElementById('answerForm');
const answerInput = document.getElementById('answerInput');
const submitBtn = document.getElementById('submitBtn');
const liveTimerText = document.getElementById('liveTimerText');
const cumulativeInfoBanner = document.getElementById('cumulativeInfoBanner');
const masteryBarsList = document.getElementById('masteryBarsList');
const takeAnotherQuizBtn = document.getElementById('takeAnotherQuizBtn');
const switchStudentBtn = document.getElementById('switchStudentBtn');
const statusBanner = document.getElementById('statusBanner');

window.addEventListener('DOMContentLoaded', () => {
  if (loginForm) {
    loginForm.addEventListener('submit', handleLoginSubmit);
  }
  if (studentIdInput) {
    studentIdInput.addEventListener('input', debounce(checkStudentSummary, 400));
  }
  if (answerForm) {
    answerForm.addEventListener('submit', handleAnswerSubmit);
  }
  if (takeAnotherQuizBtn) {
    takeAnotherQuizBtn.addEventListener('click', () => {
      if (currentStudentId) {
        startQuizForStudent(currentStudentId);
      }
    });
  }
  if (switchStudentBtn) {
    switchStudentBtn.addEventListener('click', resetToLogin);
  }
});

function debounce(func, delay) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), delay);
  };
}

function showStatus(msg, isError = false) {
  if (!statusBanner) return;
  statusBanner.textContent = msg;
  statusBanner.className = isError ? 'status-alert status-error' : 'status-alert status-success';
  statusBanner.style.display = 'block';
  setTimeout(() => {
    statusBanner.style.display = 'none';
  }, 5000);
}

function resetToLogin() {
  if (timerInterval) clearInterval(timerInterval);
  loginSection.style.display = 'block';
  quizSection.style.display = 'none';
  resultsSection.style.display = 'none';
  studentWelcomeStatus.style.display = 'none';
  studentIdInput.value = '';
  currentStudentId = null;
  currentSessionId = null;
  studentIdInput.focus();
}

async function checkStudentSummary() {
  const rawId = studentIdInput.value.trim();
  if (!rawId) {
    studentWelcomeStatus.style.display = 'none';
    return;
  }

  try {
    const res = await fetch(`/api/student/${encodeURIComponent(rawId)}/summary`);
    if (!res.ok) return;
    const data = await res.json();

    if (data.exists && data.completed_session_count > 0) {
      studentWelcomeStatus.textContent = `Welcome back! You have completed ${data.completed_session_count} session(s) (${data.total_interactions} total interactions recorded).`;
      studentWelcomeStatus.style.color = '#15803d'; // Green
      studentWelcomeStatus.style.display = 'block';
    } else {
      studentWelcomeStatus.textContent = `New student account: '${rawId}' will start Session 1.`;
      studentWelcomeStatus.style.color = '#2563eb'; // Blue
      studentWelcomeStatus.style.display = 'block';
    }
  } catch (e) {
    // Silent ignore on typing preview
  }
}

async function handleLoginSubmit(e) {
  e.preventDefault();
  const stId = studentIdInput.value.trim();
  if (!stId) return;
  currentStudentId = stId;
  await startQuizForStudent(currentStudentId);
}

async function startQuizForStudent(studentId) {
  if (timerInterval) clearInterval(timerInterval);
  loginSection.style.display = 'none';
  resultsSection.style.display = 'none';
  quizSection.style.display = 'block';
  questionText.textContent = 'Initializing adaptive assessment...';
  submitBtn.disabled = true;

  try {
    const res = await fetch('/api/session/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ student_id: studentId, source: 'live' })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Failed to start session' }));
      throw new Error(err.error || `Server responded with ${res.status}`);
    }

    const data = await res.json();
    currentSessionId = data.session_id;
    currentStudentId = data.student_id;

    activeStudentBadge.textContent = `Student: ${currentStudentId}`;

    await fetchNextQuestion();
  } catch (err) {
    console.error('Quiz start error:', err);
    showStatus(`Failed to start quiz: ${err.message}`, true);
    questionText.textContent = 'Unable to load quiz session. Please ensure the backend is running.';
  }
}

async function fetchNextQuestion() {
  const res = await fetch(`/api/session/${encodeURIComponent(currentSessionId)}/next`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'Failed to fetch next question' }));
    throw new Error(err.error || `Server responded with ${res.status}`);
  }

  const data = await res.json();

  if (data.is_complete) {
    await finishQuiz(data.unasked_predictions);
    return;
  }

  currentQuestion = data;
  renderQuestion(data);
}

function renderQuestion(q) {
  const totalQuestions = 15;
  questionProgressBadge.textContent = `Question ${q.position} of ~${totalQuestions}`;
  progressBar.style.width = `${Math.min(100, (q.position / totalQuestions) * 100)}%`;
  questionSkillTag.textContent = `Topic: ${q.skill_name} · Level: ${q.level} · ID ${q.topic}`;

  if (q.is_probe) {
    questionSkillTag.textContent = `Probe: ${q.skill_name} (not yet asked this session)`;
  }

  questionText.textContent = q.question_text;

  // Show pre-answer prediction, but no correct/incorrect verdict.
  if (q.predicted_p !== null && q.predicted_p !== undefined) {
    predictedPText.textContent = `${Math.round(q.predicted_p * 100)}%`;
    predictionBanner.style.display = 'block';
  } else {
    predictionBanner.style.display = 'none';
  }

  answerInput.value = '';
  answerInput.disabled = false;
  submitBtn.disabled = false;
  answerInput.focus();

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

  if (timerInterval) clearInterval(timerInterval);
  const responseTimeMs = Math.round(performance.now() - questionStartTime);

  submitBtn.disabled = true;
  answerInput.disabled = true;

  try {
    const res = await fetch(`/api/session/${encodeURIComponent(currentSessionId)}/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        submitted_answer: submittedAnswer,
        response_time_ms: responseTimeMs,
        hint_used: 0
      })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Error submitting answer' }));
      throw new Error(err.error || `Server responded with ${res.status}`);
    }

    // No correct/incorrect verdict is shown; go straight to the next question.
    await fetchNextQuestion();
  } catch (err) {
    console.error('Answer submit error:', err);
    showStatus(`Submission error: ${err.message}`, true);
    submitBtn.disabled = false;
    answerInput.disabled = false;
  }
}

async function finishQuiz(unaskedPredictions) {
  if (timerInterval) clearInterval(timerInterval);
  quizSection.style.display = 'none';
  resultsSection.style.display = 'block';
  masteryBarsList.innerHTML = '<p style="color: var(--text-muted);">Estimating cumulative per-skill mastery via GatedKT...</p>';

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
    const totalProcessed = data.num_interactions_processed || 0;
    const skillCounts = data.skill_interaction_counts || {};
    const lowConfidence = data.low_confidence_skills || [];

    const summaryRes = await fetch(`/api/student/${encodeURIComponent(currentStudentId)}/summary`);
    let sessionCount = 1;
    if (summaryRes.ok) {
      const summaryData = await summaryRes.json();
      sessionCount = summaryData.completed_session_count || summaryData.session_count || 1;
    }

    cumulativeInfoBanner.textContent = `Mastery estimate based on ${totalProcessed} total interaction(s) across ${sessionCount} session(s) for student '${currentStudentId}'.`;

    takeAnotherQuizBtn.textContent = `Take Another Quiz (Continue as '${currentStudentId}')`;

    renderMasteryBars(mastery, skillCounts, lowConfidence, data.mastery_summary);
    if (unaskedPredictions) {
      renderUnaskedPredictions(unaskedPredictions);
    }
    renderSkillGraph(currentStudentId, currentSessionId);
    renderMasteryTrend(currentStudentId);
  } catch (err) {
    console.error('Mastery prediction error:', err);
    showStatus(`Prediction error: ${err.message}`, true);
    masteryBarsList.innerHTML = '<p style="color: var(--accent-rose);">Failed to calculate mastery predictions.</p>';
  }
}

function renderUnaskedPredictions(unaskedPredictions) {
  let container = document.getElementById('unaskedPredictionsCard');
  if (!container) {
    container = document.createElement('div');
    container.id = 'unaskedPredictionsCard';
    container.style.cssText = 'margin-top: 18px; background: #f0f9ff; border: 1px solid var(--border-color); border-left: 4px solid #0891b2; border-radius: 0 8px 8px 0; padding: 14px 18px;';
    masteryBarsList.parentNode.insertBefore(container, masteryBarsList.nextSibling);
  }
  const nameById = {};
  SKILL_SEQUENCE.forEach(s => { nameById[s.raw_skill_id] = s.skill_name; });

  const rows = Object.entries(unaskedPredictions).map(([id, p]) => {
    const name = nameById[Number(id)] || `Skill ${id}`;
    return `<div style="margin-bottom: 4px;">• ${name} (ID ${id}): <strong>${Math.round(p * 100)}%</strong> predicted mastery via graph propagation</div>`;
  }).join('');

  container.innerHTML = `
    <div style="font-weight: 700; font-size: 0.92rem; color: #155e75; margin-bottom: 4px;">🧠 Graph-propagated predictions for unasked topics</div>
    <div style="font-size: 0.85rem; color: #0e7490; line-height: 1.5;">${rows || 'All topics were covered.'}</div>
  `;
}

let activeNetworkInstance = null;
let activeTrendChart = null;

function createNodeTooltip(n, pct, isWeak, isLowData) {
  const container = document.createElement('div');
  container.style.padding = '8px 12px';
  container.style.fontSize = '12px';
  container.style.lineHeight = '1.45';
  container.style.color = '#0f172a';
  container.style.background = '#ffffff';
  container.style.borderRadius = '6px';
  container.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
  container.style.border = '1px solid #e2e8f0';

  const strongTitle = document.createElement('div');
  strongTitle.style.fontWeight = '700';
  strongTitle.style.color = '#1e293b';
  strongTitle.textContent = `${n.name} (ID: ${n.raw_skill_id})`;
  container.appendChild(strongTitle);

  const masteryDiv = document.createElement('div');
  masteryDiv.innerHTML = `<strong>Mastery:</strong> ${pct}%`;
  container.appendChild(masteryDiv);

  const interDiv = document.createElement('div');
  interDiv.innerHTML = `<strong>Interactions:</strong> ${n.interaction_count || 1}`;
  container.appendChild(interDiv);

  if (isWeak) {
    const weakDiv = document.createElement('div');
    weakDiv.style.color = '#e11d48';
    weakDiv.style.fontWeight = '600';
    weakDiv.style.marginTop = '4px';
    weakDiv.textContent = '⚠️ Weak Skill (<50%)';
    container.appendChild(weakDiv);
  }

  if (isLowData) {
    const lowDiv = document.createElement('div');
    lowDiv.style.color = '#d97706';
    lowDiv.style.fontWeight = '600';
    lowDiv.style.marginTop = '2px';
    lowDiv.textContent = '⚠️ Low Data (<3 answers)';
    container.appendChild(lowDiv);
  }

  return container;
}

function createEdgeTooltip(w, srcName, tgtName) {
  const container = document.createElement('div');
  container.style.padding = '8px 12px';
  container.style.fontSize = '12px';
  container.style.lineHeight = '1.4';
  container.style.color = '#1e293b';
  container.style.background = '#ffffff';
  container.style.borderRadius = '6px';
  container.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
  container.style.border = '1px solid #e2e8f0';

  const titleEl = document.createElement('div');
  titleEl.style.fontWeight = 'bold';
  titleEl.style.fontSize = '13px';
  titleEl.style.marginBottom = '4px';
  titleEl.style.color = '#0f172a';
  titleEl.textContent = 'Curriculum Relation';
  container.appendChild(titleEl);

  if (srcName && tgtName) {
    const pairRow = document.createElement('div');
    pairRow.textContent = `${srcName} ↔ ${tgtName}`;
    pairRow.style.fontWeight = '600';
    pairRow.style.marginBottom = '2px';
    container.appendChild(pairRow);
  }

  const weightRow = document.createElement('div');
  const val = (w !== undefined && w !== null) ? Number(w).toFixed(2) : '0.50';
  weightRow.innerHTML = `<strong>PPMI Co-occurrence Weight:</strong> ${val}`;
  container.appendChild(weightRow);

  return container;
}

function renderMasteryBars(mastery, skillCounts = {}, lowConfidence = [], masterySummary = '') {
  masteryBarsList.innerHTML = '';

  // Render bars for all 10 skills in exact fixed skill_sequence order
  SKILL_SEQUENCE.forEach(q => {
    const rawIdStr = String(q.raw_skill_id);
    const score = mastery[rawIdStr] !== undefined ? mastery[rawIdStr] : 0.5;
    const pct = Math.round(score * 100);
    const intCount = skillCounts[rawIdStr] !== undefined ? skillCounts[rawIdStr] : 1;
    const isLowData = intCount < 3 || lowConfidence.includes(rawIdStr);

    let barColor = '#10b981'; // Green (high mastery)
    if (pct < 50) {
      barColor = '#f43f5e'; // Rose/Red (low mastery)
    } else if (pct < 75) {
      barColor = '#f59e0b'; // Amber (moderate mastery)
    }

    const dataChipHtml = isLowData
      ? `<span class="low-data-chip" title="Low data: based on ${intCount} answer(s). Minimum 3 recommended for high confidence.">⚠️ ${intCount} answer${intCount === 1 ? '' : 's'} (Low Data)</span>`
      : `<span class="high-data-chip" title="Well supported: based on ${intCount} interaction(s)">${intCount} answers</span>`;

    const item = document.createElement('div');
    item.className = 'mastery-item';
    if (isLowData) {
      item.style.opacity = '0.92';
      item.style.borderLeft = '3px dashed #f59e0b';
    }

    item.innerHTML = `
      <div class="mastery-meta" style="align-items: center;">
        <div>
          <span>${q.position}. ${q.skill_name} <span style="color: var(--text-muted); font-size: 0.8rem;">(ID ${q.raw_skill_id})</span></span>
          <span style="margin-left: 8px;">${dataChipHtml}</span>
        </div>
        <span style="color: ${barColor}; font-weight: 700; font-size: 1.05rem;">${pct}%</span>
      </div>
      <div class="mastery-bar-bg" style="${isLowData ? 'border: 1px dashed #cbd5e1;' : ''}">
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

  // Task 19: Render auto-generated synthesis summary below mastery bars
  const summaryCard = document.getElementById('masterySummaryCard');
  const summaryText = document.getElementById('masterySummaryText');
  if (summaryCard && summaryText) {
    if (masterySummary && masterySummary.trim().length > 0) {
      summaryText.textContent = masterySummary;
      summaryCard.style.display = 'block';
    } else {
      summaryCard.style.display = 'none';
    }
  }
}

async function renderSkillGraph(studentId, sessionId) {
  const container = document.getElementById('skillGraphVis');
  const emptyState = document.getElementById('skillGraphEmptyState');
  const graphContainer = document.getElementById('skillGraphContainer');
  if (!container) return;

  try {
    const url = `/api/graph/subgraph?student_id=${encodeURIComponent(studentId || '')}&session_id=${encodeURIComponent(sessionId || '')}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const { nodes, edges, has_completed_session, has_history } = data;

    // Honest empty state handling if new student with no completed session
    if (studentId && has_completed_session === false && !has_history) {
      if (emptyState) emptyState.style.display = 'block';
      if (graphContainer) graphContainer.style.display = 'none';
      return;
    }

    if (emptyState) emptyState.style.display = 'none';
    if (graphContainer) graphContainer.style.display = 'block';

    // Format nodes for vis-network with HTML element tooltips (fixing raw <br> tags)
    const visNodes = (nodes || []).map((n, idx) => {
      const masteryVal = n.mastery !== undefined ? n.mastery : 0.5;
      const pct = Math.round(masteryVal * 100);
      const isWeak = masteryVal < 0.50;
      const isLowData = n.is_low_data || (n.interaction_count !== undefined && n.interaction_count < 3);

      let nodeBg = '#10b981'; // Green
      let borderCol = '#059669';
      if (isWeak) {
        nodeBg = '#f43f5e'; // Red/Rose
        borderCol = '#be123c';
      } else if (pct < 75) {
        nodeBg = '#f59e0b'; // Amber
        borderCol = '#d97706';
      }

      const shortLabel = n.name.length > 18 ? n.name.slice(0, 16) + '…' : n.name;
      const dataFlag = isLowData ? `\n(⚠️ ${n.interaction_count || 1} ans)` : `\n(${n.interaction_count || 1} ans)`;

      return {
        id: n.raw_skill_id,
        label: `#${n.position} ${shortLabel}\n${pct}%${dataFlag}`,
        title: createNodeTooltip(n, pct, isWeak, isLowData),
        shape: 'box',
        margin: 10,
        color: {
          background: nodeBg,
          border: isLowData ? '#f59e0b' : borderCol,
          highlight: {
            background: nodeBg,
            border: '#0f172a'
          }
        },
        font: {
          color: '#ffffff',
          size: 11,
          face: 'sans-serif',
          bold: { mod: 'bold' }
        },
        borderWidth: isLowData ? 3 : 2,
        shapeProperties: {
          borderDashes: isLowData ? [4, 4] : false,
          borderRadius: 8
        },
        opacity: isLowData ? 0.88 : 1.0
      };
    });

    // Format edges for vis-network with DOM element tooltips
    const nodeNameMap = {};
    (nodes || []).forEach(n => {
      nodeNameMap[n.raw_skill_id] = n.name;
    });

    const visEdges = (edges || []).map(e => {
      const w = e.weight || 0.5;
      const width = Math.min(6, Math.max(1.2, (w / 2.5) * 2));
      const opacity = Math.min(0.95, Math.max(0.25, w / 9));
      const srcName = nodeNameMap[e.source] || `Skill #${e.source}`;
      const tgtName = nodeNameMap[e.target] || `Skill #${e.target}`;

      return {
        from: e.source,
        to: e.target,
        value: w,
        width: width,
        color: {
          color: '#94a3b8',
          opacity: opacity,
          highlight: '#3b82f6'
        },
        title: createEdgeTooltip(w, srcName, tgtName)
      };
    });

    if (window.vis && window.vis.Network) {
      if (activeNetworkInstance) {
        activeNetworkInstance.destroy();
      }

      const graphData = {
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges)
      };

      const options = {
        physics: {
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {
            gravitationalConstant: -40,
            centralGravity: 0.015,
            springLength: 100,
            springConstant: 0.08
          },
          stabilization: { iterations: 120 }
        },
        interaction: {
          hover: true,
          tooltipDelay: 100,
          zoomView: true
        }
      };

      activeNetworkInstance = new vis.Network(container, graphData, options);
    }

    // Task 16: Render weak-skill relational callouts
    const calloutsCard = document.getElementById('weakRelationalCallouts');
    const calloutsList = document.getElementById('weakRelationalCalloutsList');
    const callouts = data.weak_relational_callouts || [];

    if (calloutsCard && calloutsList) {
      if (callouts.length > 0) {
        calloutsList.innerHTML = callouts.map(c => `<div style="margin-bottom: 4px;">• ${c}</div>`).join('');
        calloutsCard.style.display = 'block';
      } else {
        calloutsCard.style.display = 'none';
      }
    }
  } catch (err) {
    console.warn('Could not render skill relation graph:', err);
  }
}

async function renderMasteryTrend(studentId) {
  const canvas = document.getElementById('trendChartCanvas');
  const badge = document.getElementById('trendSessionBadge');
  const note = document.getElementById('trendSingleSessionNote');
  const skillSelect = document.getElementById('trendSkillSelect');
  if (!canvas || !studentId) return;

  try {
    const res = await fetch(`/api/student/${encodeURIComponent(studentId)}/trend`);
    if (!res.ok) return;
    const data = await res.json();
    const trend = data.trend || [];

    if (badge) badge.textContent = `${trend.length} Session(s) Recorded`;

    if (trend.length <= 1) {
      if (note) note.style.display = 'block';
    } else {
      if (note) note.style.display = 'none';
    }

    if (!window.Chart) return;

    // Palette for 10 distinct skills
    const colors = [
      '#2563eb', '#0891b2', '#059669', '#16a34a',
      '#65a30d', '#d97706', '#ea580c', '#e11d48',
      '#9333ea', '#4f46e5'
    ];

    const labels = trend.map((t, i) => `Session ${t.session_index || i + 1}`);

    // Populate skill select dropdown once
    if (skillSelect && skillSelect.options.length <= 1 && SKILL_SEQUENCE.length > 0) {
      SKILL_SEQUENCE.forEach((q, idx) => {
        const opt = document.createElement('option');
        opt.value = String(q.raw_skill_id);
        opt.textContent = `${q.position}. ${q.skill_name}`;
        skillSelect.appendChild(opt);
      });

      skillSelect.onchange = () => {
        renderMasteryTrend(studentId);
      };
    }

    const selectedSkill = skillSelect ? skillSelect.value : 'all';

    const datasets = [];
    SKILL_SEQUENCE.forEach((q, idx) => {
      const rawIdStr = String(q.raw_skill_id);
      if (selectedSkill !== 'all' && selectedSkill !== rawIdStr) {
        return;
      }

      const points = trend.map(t => {
        const val = (t.mastery && t.mastery[rawIdStr] !== undefined) ? t.mastery[rawIdStr] : 0.5;
        return Math.round(val * 100);
      });

      datasets.push({
        label: `${q.position}. ${q.skill_name}`,
        data: points,
        borderColor: colors[idx % colors.length],
        backgroundColor: colors[idx % colors.length],
        borderWidth: 2.5,
        tension: 0.15,
        pointRadius: 5,
        pointHoverRadius: 7
      });
    });

    if (activeTrendChart) {
      activeTrendChart.destroy();
    }

    const ctx = canvas.getContext('2d');
    activeTrendChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: 'Estimated Mastery (%)' },
            ticks: { callback: v => `${v}%` }
          },
          x: {
            title: { display: true, text: 'Historical Assessment Sequence' }
          }
        },
        plugins: {
          legend: {
            display: selectedSkill === 'all' && datasets.length <= 5,
            position: 'bottom'
          },
          tooltip: {
            callbacks: {
              label: item => `${item.dataset.label}: ${item.raw}% mastery`
            }
          }
        }
      }
    });

  } catch (err) {
    console.warn('Could not render mastery trend chart:', err);
  }
}

async function renderGateDynamics(sessionId) {
  const container = document.getElementById('gateStepsContainer');
  const activeContent = document.getElementById('gateActiveContent');
  const insufficientNote = document.getElementById('gateInsufficientNote');
  const plotImg = document.getElementById('gatePlotImg');
  const plotLoading = document.getElementById('gatePlotLoading');
  if (!container || !sessionId) return;

  try {
    const res = await fetch(`/api/session/${encodeURIComponent(sessionId)}/gate_detail`);
    if (!res.ok) return;
    const data = await res.json();

    if (!data.available) {
      if (insufficientNote) insufficientNote.style.display = 'block';
      if (activeContent) activeContent.style.display = 'none';
      return;
    }

    if (insufficientNote) insufficientNote.style.display = 'none';
    if (activeContent) activeContent.style.display = 'block';

    const means = data.step_gate_means || [];
    container.innerHTML = '';
    means.forEach((val, idx) => {
      const stepNum = idx + 1;
      const pct = Math.round(val * 100);

      let tagColor = '#0284c7';
      let focusLabel = 'Short-Term GRU Focus';
      if (val < 0.45) {
        tagColor = '#6366f1';
        focusLabel = 'Long-Term & PPMI Graph';
      } else if (val <= 0.55) {
        tagColor = '#64748b';
        focusLabel = 'Balanced Gate Fusion';
      }

      const chip = document.createElement('div');
      chip.style.cssText = 'background: white; border: 1px solid var(--border-color); border-radius: 6px; padding: 6px 10px; font-size: 0.78rem; display: flex; flex-direction: column; gap: 2px; min-width: 95px; text-align: center;';
      chip.innerHTML = `
        <div style="font-weight: 700; color: var(--text-primary);">Step ${stepNum} → ${stepNum + 1}</div>
        <div style="font-weight: 800; color: ${tagColor}; font-size: 0.95rem;">${pct}%</div>
        <div style="font-size: 0.68rem; color: var(--text-muted);">${focusLabel}</div>
      `;
      chip.title = `Step ${stepNum}: Gate mean = ${val.toFixed(4)}. Higher values place greater weight on the short-term encoder; lower values rely more on long-term GRU and GAT graph embeddings.`;
      container.appendChild(chip);
    });

    // Load gate plot image
    if (plotImg) {
      plotImg.src = `/api/session/${encodeURIComponent(sessionId)}/gate_plot.png?t=${Date.now()}`;
      plotImg.onload = () => {
        plotImg.style.display = 'inline-block';
        if (plotLoading) plotLoading.style.display = 'none';
      };
      plotImg.onerror = () => {
        if (plotLoading) plotLoading.textContent = 'Gate plot generation not available for this session length.';
      };
    }

  } catch (err) {
    console.warn('Could not render gate dynamics:', err);
  }
}

