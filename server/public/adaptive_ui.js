/**
 * server/public/adaptive_ui.js
 * ----------------------------
 * adaptive-v2 front-end additions (quiz screen + results screen + landing).
 * Loaded BEFORE app.js; app.js calls into window.GKT through small hooks that
 * are each wrapped in try/catch, so a failure here can never break the
 * existing quiz flow. Every new field from the API is feature-detected.
 *
 * Rules:
 * - SHOW_VERDICT = false: no correct/incorrect verdict is shown during the quiz.
 * - SHOW_ADAPTATION_TOAST = true: a neutral "Difficulty adjusted" toast when a
 *   topic's level changes. NOTE: any visible level change implicitly hints at
 *   whether the previous answer on that topic was right; this is accepted.
 * - One band function (band()) drives every colour: prediction badge, mastery
 *   bars, result cards, graph nodes and the dashboard heatmap.
 */
(function () {
  'use strict';

  const BAND_HIGH = 0.70;
  const BAND_MID = 0.40;
  const SHOW_VERDICT = false;          // never show correct/incorrect during the quiz
  const SHOW_ADAPTATION_TOAST = true;  // neutral wording; level changes implicitly hint at correctness
  const DEFAULT_TOTAL = 15;
  const LEVELS = ['easy', 'medium', 'hard'];
  const LEVEL_LABEL = { easy: 'EASY', medium: 'MEDIUM', hard: 'HARD' };
  const REDUCED_MOTION = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const COLORS = {
    good: { fg: '#16a34a', bg: '#dcfce7', text: '#166534' },
    mid: { fg: '#d97706', bg: '#fef3c7', text: '#92400e' },
    bad: { fg: '#dc2626', bg: '#fee2e2', text: '#991b1b' },
    neutral: { fg: '#64748b', bg: '#f1f5f9', text: '#334155' }
  };

  /** The ONE band function used everywhere. p in [0,1] (or null). */
  function band(p) {
    if (p === null || p === undefined || isNaN(p)) {
      return { key: 'neutral', icon: '•', label: 'No prediction', ...COLORS.neutral };
    }
    if (p >= BAND_HIGH) return { key: 'good', icon: '✓', label: 'Likely to get this right', ...COLORS.good };
    if (p >= BAND_MID) return { key: 'mid', icon: '≈', label: 'Toss-up', ...COLORS.mid };
    return { key: 'bad', icon: '!', label: 'Challenging for you right now', ...COLORS.bad };
  }

  const LEVEL_BAND = { easy: 'good', medium: 'mid', hard: 'bad' };
  function levelColors(level) { return COLORS[LEVEL_BAND[level]] || COLORS.neutral; }

  const STATUS_META = {
    mastered: { label: 'Mastered', icon: '★', band: 'good' },
    proficient: { label: 'Proficient', icon: '✓', band: 'good' },
    developing: { label: 'Developing', icon: '◐', band: 'mid' },
    needs_support: { label: 'Needs support', icon: '!', band: 'bad' },
    not_enough_evidence: { label: 'Not enough evidence', icon: '?', band: 'neutral' }
  };
  function statusMeta(s) {
    const m = STATUS_META[s] || { label: s || 'Unknown', icon: '?', band: 'neutral' };
    return { ...m, ...COLORS[m.band] };
  }

  function esc(s) {
    return String(s === undefined || s === null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function pct(p) { return `${Math.round(p * 100)}%`; }
  function $(id) { return document.getElementById(id); }
  function safe(name, fn) {
    return function (...args) {
      try { return fn.apply(this, args); } catch (e) { console.warn(`[GKT] ${name} failed:`, e); return undefined; }
    };
  }

  // ------------------------------------------------------------------
  // Session state (client-side log, collected from /next and /answer)
  // ------------------------------------------------------------------
  const state = {
    sessionLog: [],          // [{position, topic, skill_name, level, predicted_p, correct, is_probe}]
    lastLevelByTopic: {},    // topic -> last level served
    names: {},               // topic -> skill_name (from API responses)
    totalQuestions: DEFAULT_TOTAL,
    currentQuestion: null,
    policy: null,
    levelChart: null,
    predChart: null,
    sessionId: null
  };

  function resetSession(sessionId) {
    state.sessionLog = [];
    state.lastLevelByTopic = {};
    state.totalQuestions = DEFAULT_TOTAL;
    state.currentQuestion = null;
    state.policy = null;
    state.sessionId = sessionId || null;
    const seg = $('levelSegments'); if (seg) seg.innerHTML = '';
    const reason = $('decisionReason'); if (reason) { reason.textContent = ''; reason.style.display = 'none'; }
    if (state.levelChart) { state.levelChart.destroy(); state.levelChart = null; }
    const v2 = $('resultsV2'); if (v2) v2.innerHTML = '';
  }

  // ------------------------------------------------------------------
  // T4: quiz screen
  // ------------------------------------------------------------------
  function onQuestion(q) {
    state.currentQuestion = q;
    if (q.skill_name) state.names[q.topic] = q.skill_name;
    if (Number.isInteger(q.total_questions) && q.total_questions > 0) state.totalQuestions = q.total_questions;
    if (q.policy) state.policy = q.policy;
    const prevLevel = state.lastLevelByTopic[q.topic];
    state.sessionLog.push({
      position: q.position, topic: q.topic, skill_name: q.skill_name, level: q.level,
      predicted_p: (q.predicted_p === undefined ? null : q.predicted_p), correct: null, is_probe: !!q.is_probe
    });
    state.lastLevelByTopic[q.topic] = q.level;

    renderHeader(q);
    renderPredictionCard(q);
    renderDifficultyPill(q, prevLevel);
    renderSegments();
    renderAdaptPanel(q);
    setAdapting(false);
  }

  const renderHeader = safe('renderHeader', function (q) {
    const badge = $('questionProgressBadge');
    if (badge) badge.textContent = `Question ${q.position} of ${state.totalQuestions}`;
    const bar = $('progressBar');
    if (bar) bar.style.width = `${Math.min(100, (q.position / state.totalQuestions) * 100)}%`;
    const tag = $('questionSkillTag');
    if (tag) {
      tag.textContent = q.is_probe ? `Probe: ${q.skill_name}` : q.skill_name;
      tag.title = `Topic ID ${q.topic}`;
      tag.classList.add('topic-chip');
    }
  });

  let countUpTimer = null;
  const renderPredictionCard = safe('renderPredictionCard', function (q) {
    const card = $('predictionCard');
    if (!card) return;
    const banner = $('predictionBanner');
    if (banner) banner.style.display = 'none'; // superseded by the card (kept in DOM for compatibility)
    card.style.display = 'flex';
    const p = q.predicted_p;
    const valEl = $('predCardValue');
    const labelEl = $('predCardLabel');
    const gauge = $('predGaugeArc');
    const subtitle = $('predCardSubtitle');
    if (countUpTimer) { clearInterval(countUpTimer); countUpTimer = null; }

    if (p === null || p === undefined) {
      const isProbe = !!q.is_probe;
      card.className = 'pred-card band-neutral';
      valEl.textContent = '—';
      labelEl.innerHTML = isProbe
        ? '<span aria-hidden="true">◇</span> Probe question: checking a new topic'
        : '<span aria-hidden="true">◇</span> Warm-up: no prediction yet (the model needs 2 answers)';
      if (gauge) gauge.style.strokeDashoffset = '100';
      subtitle.textContent = isProbe ? 'Prediction for a topic not asked yet' : 'Predicted by GatedKT before you answer';
      return;
    }
    const b = band(p);
    card.className = `pred-card band-${b.key}`;
    labelEl.innerHTML = `<span aria-hidden="true">${b.icon}</span> ${esc(b.label)}`;
    subtitle.textContent = 'Predicted by GatedKT before you answer';
    const target = Math.round(p * 100);
    if (gauge) gauge.style.strokeDashoffset = String(100 - target);
    if (REDUCED_MOTION) { valEl.textContent = `${target}%`; return; }
    let cur = 0;
    const step = Math.max(1, Math.round(target / 20));
    valEl.textContent = '0%';
    countUpTimer = setInterval(() => {
      cur = Math.min(target, cur + step);
      valEl.textContent = `${cur}%`;
      if (cur >= target) { clearInterval(countUpTimer); countUpTimer = null; }
    }, 25);
  });

  const renderDifficultyPill = safe('renderDifficultyPill', function (q, prevLevel) {
    const pill = $('difficultyPill');
    if (!pill) return;
    const lvl = LEVELS.includes(q.level) ? q.level : 'medium';
    const idx = LEVELS.indexOf(lvl);
    const c = levelColors(lvl);
    pill.className = `difficulty-pill level-${lvl}`;
    pill.style.color = c.text;
    pill.style.background = c.bg;
    pill.style.borderColor = c.fg;
    const steps = [0, 1, 2].map(i =>
      `<span class="ladder-step${i <= idx ? ' on' : ''}" style="height:${6 + i * 4}px;${i <= idx ? `background:${c.fg}` : ''}"></span>`).join('');
    pill.innerHTML = `<span class="ladder" aria-hidden="true">${steps}</span><span>${LEVEL_LABEL[lvl]}</span>`;
    pill.setAttribute('aria-label', `Difficulty: ${lvl}`);
    if (prevLevel && prevLevel !== lvl) {
      if (!REDUCED_MOTION) {
        pill.classList.remove('pill-bump'); void pill.offsetWidth; pill.classList.add('pill-bump');
      }
      if (SHOW_ADAPTATION_TOAST) showToast('Difficulty adjusted for this topic');
    }
  });

  function showToast(msg) {
    const t = $('adaptToast');
    if (!t) return;
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(showToast._h);
    showToast._h = setTimeout(() => t.classList.remove('show'), 2200);
  }

  const renderSegments = safe('renderSegments', function () {
    const seg = $('levelSegments');
    if (!seg) return;
    const total = Math.max(state.totalQuestions, state.sessionLog.length);
    let html = '';
    for (let i = 0; i < total; i++) {
      const e = state.sessionLog[i];
      if (e) {
        const c = levelColors(e.level);
        const cur = i === state.sessionLog.length - 1;
        html += `<span class="seg${cur ? ' current' : ''}" style="background:${c.fg}" title="Q${e.position}: ${esc(e.level)}${e.is_probe ? ' (probe)' : ''}"></span>`;
      } else {
        html += '<span class="seg empty"></span>';
      }
    }
    seg.innerHTML = html;
    seg.setAttribute('aria-label', `Questions served so far coloured by difficulty level: ${state.sessionLog.map(e => e.level).join(', ')}`);
  });

  const POLICY_LABEL = { model: 'Model-guided difficulty', staircase: 'Staircase difficulty' };

  const renderAdaptPanel = safe('renderAdaptPanel', function (q) {
    const panel = $('adaptPanel');
    if (!panel) return;
    panel.style.display = 'block';
    const pb = $('policyBadge');
    if (pb) {
      const label = POLICY_LABEL[q.policy || state.policy];
      pb.textContent = label || '';
      pb.style.display = label ? 'inline-flex' : 'none';
    }
    const reasonEl = $('decisionReason');
    const reason = q.decision && typeof q.decision.reason === 'string' ? q.decision.reason : '';
    if (reasonEl) { reasonEl.textContent = reason; reasonEl.style.display = reason ? 'block' : 'none'; }
    renderTopicChips(q);
    renderLevelChart();
  });

  const renderTopicChips = safe('renderTopicChips', function (q) {
    const box = $('topicChips');
    if (!box) return;
    const ids = [];
    const add = t => { const n = Number(t); if (!ids.includes(n)) ids.push(n); };
    state.sessionLog.forEach(e => add(e.topic));
    if (q.decision && q.decision.topic_scores) Object.keys(q.decision.topic_scores).forEach(add);
    box.innerHTML = ids.map(t => {
      const lvl = state.lastLevelByTopic[t];
      const c = lvl ? levelColors(lvl) : COLORS.neutral;
      const name = state.names[t] || (window.SKILL_NAME_BY_ID && window.SKILL_NAME_BY_ID[t]) || `Topic ${t}`;
      return `<span class="topic-level-chip" style="border-color:${c.fg};background:${c.bg};color:${c.text}" title="Topic ID ${t}">${esc(name)} · ${lvl ? esc(lvl) : 'not asked yet'}</span>`;
    }).join('');
  });

  const renderLevelChart = safe('renderLevelChart', function () {
    const canvas = $('levelStepChart');
    if (!canvas || !window.Chart) return;
    const log = state.sessionLog;
    const palette = ['#2563eb', '#0891b2', '#059669', '#7c3aed', '#db2777', '#ea580c', '#4f46e5', '#65a30d', '#0f766e', '#9333ea'];
    const topicOrder = [];
    log.forEach(e => { if (!topicOrder.includes(e.topic)) topicOrder.push(e.topic); });
    const data = log.map(e => LEVELS.indexOf(e.level) + 1);
    const pointColors = log.map(e => palette[topicOrder.indexOf(e.topic) % palette.length]);
    const cfg = {
      type: 'line',
      data: {
        labels: log.map(e => `Q${e.position}`),
        datasets: [{
          label: 'Difficulty served', data, stepped: true, borderColor: '#94a3b8', borderWidth: 2,
          pointBackgroundColor: pointColors, pointBorderColor: pointColors, pointRadius: 5, fill: false, clip: 8
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: REDUCED_MOTION ? false : { duration: 250 },
        layout: { padding: { top: 8, bottom: 4 } },
        scales: {
          y: { min: 1, max: 3, ticks: { stepSize: 1, callback: v => ({ 1: 'Easy', 2: 'Medium', 3: 'Hard' }[v] || '') } },
          x: { ticks: { maxRotation: 0, autoSkip: true } }
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => { const e = log[ctx.dataIndex]; return `${e.skill_name}: ${e.level}${e.is_probe ? ' (probe)' : ''}`; } } }
        }
      }
    };
    if (state.levelChart) {
      state.levelChart.data = cfg.data;
      state.levelChart.update();
    } else {
      state.levelChart = new Chart(canvas.getContext('2d'), cfg);
    }
  });

  const setAdapting = safe('setAdapting', function (on) {
    const sh = $('adaptingShimmer');
    if (sh) sh.style.display = on ? 'flex' : 'none';
  });

  function onSubmitStart() { setAdapting(true); }

  function onAnswer(a) {
    // Stored for the results screen only; never displayed during the quiz (SHOW_VERDICT=false).
    try {
      const e = state.sessionLog.find(x => x.position === a.position) || state.sessionLog[state.sessionLog.length - 1];
      if (e && (a.correct === 0 || a.correct === 1)) e.correct = a.correct;
      if (SHOW_VERDICT) { /* intentionally unused */ }
    } catch (err) { console.warn('[GKT] onAnswer failed', err); }
  }

  function onSubmitError() { setAdapting(false); }

  // ------------------------------------------------------------------
  // T5: results screen
  // ------------------------------------------------------------------
  async function fetchJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  function accuracyOf(t) {
    let n = 0, c = 0;
    LEVELS.forEach(l => { const x = (t.correct_by_level || {})[l]; if (x) { n += x.n; c += x.correct; } });
    return n ? c / n : null;
  }

  function statCard(label, value, sub) {
    return `<div class="stat-card"><div class="stat-label">${esc(label)}</div><div class="stat-value">${esc(value)}</div>${sub ? `<div class="stat-sub">${esc(sub)}</div>` : ''}</div>`;
  }

  function topicCard(t) {
    const m = statusMeta(t.status);
    const conf = t.n_answers < 3 ? 'low confidence' : (t.n_answers < 5 ? 'some evidence' : 'more evidence');
    const levels = (t.levels_attempted || []).map(l => `<span class="mini-level" style="background:${levelColors(l).bg};color:${levelColors(l).text}">${esc(l)}</span>`).join(' ');
    return `<div class="topic-card" style="border-left-color:${m.fg}">
      <div class="topic-card-head">
        <span class="topic-card-name" title="Topic ID ${esc(t.topic)}">${esc(t.skill_name || `Topic ${t.topic}`)}</span>
        <span class="status-chip" style="background:${m.bg};color:${m.text};border-color:${m.fg}"><span aria-hidden="true">${m.icon}</span> ${esc(m.label)}</span>
      </div>
      <div class="topic-card-meta">
        <span class="evidence-chip">${t.n_answers} answer${t.n_answers === 1 ? '' : 's'} - ${conf}</span>
        ${levels ? `<span class="levels-tried">Levels tried: ${levels}</span>` : ''}
        ${t.ready_for_next ? '<span class="ready-badge"><span aria-hidden="true">↑</span> Ready for harder questions</span>' : ''}
      </div>
      <div class="topic-card-action">Suggested next step (rule-based): <strong>${esc(t.recommended_action || '')}</strong></div>
      ${t.evidence_note ? `<div class="topic-card-note">${esc(t.evidence_note)}</div>` : ''}
    </div>`;
  }

  function difficultyProfile(topics, reliable) {
    const rows = topics.map(t => {
      const cbl = t.correct_by_level || {};
      const bars = LEVELS.map(l => {
        let v, label;
        if (reliable && t.p) { v = t.p[l]; label = v === null || v === undefined ? '—' : pct(v); }
        else { const x = cbl[l]; v = x && x.n ? x.correct / x.n : null; label = x && x.n ? `${x.correct}/${x.n}` : '—'; }
        const b = band(v);
        const answered = cbl[l] && cbl[l].n > 0;
        return `<div class="dp-bar-wrap" title="${esc(l)}: ${label}${answered ? ' (answered at this level)' : ''}">
          <div class="dp-bar-track"><div class="dp-target" style="bottom:70%"></div>
            <div class="dp-bar" style="height:${v === null || v === undefined ? 0 : Math.round(v * 100)}%;background:${b.fg}"></div></div>
          <div class="dp-bar-label">${l[0].toUpperCase()}${answered ? '<span class="dp-dot" aria-label="answered">●</span>' : ''}</div>
          <div class="dp-bar-val">${label}</div></div>`;
      }).join('');
      return `<div class="dp-topic"><div class="dp-name" title="Topic ID ${esc(t.topic)}">${esc(t.skill_name || t.topic)}</div><div class="dp-bars">${bars}</div></div>`;
    }).join('');
    const note = reliable
      ? 'Bars: predicted chance of success at Easy / Medium / Hard (model estimate). Dashed line: 70% (proficiency threshold). ● = you answered at that level.'
      : 'Showing observed accuracy by level (correct/answered) because the model-based difficulty profile is not verified for this setup. ● = answered at that level.';
    return `<div class="dp-grid">${rows}</div><p class="small-note">${esc(note)}</p>`;
  }

  function measuredVsPredicted(topics, unasked, reliable) {
    const measured = topics.filter(t => t.n_this_session > 0 || t.n_answers > 0);
    const row = (name, id, v, hatched, sub) => {
      const b = band(v);
      return `<div class="mvp-row"><div class="mvp-name" title="Topic ID ${esc(id)}">${esc(name)}</div>
        <div class="mvp-track"><div class="mvp-fill${hatched ? ' hatched' : ''}" style="width:${v === null || v === undefined ? 0 : Math.round(v * 100)}%;--bar:${b.fg}"></div></div>
        <div class="mvp-val">${v === null || v === undefined ? '—' : pct(v)}${sub ? ` <span class="mvp-sub">${esc(sub)}</span>` : ''}</div></div>`;
    };
    const mRows = measured.map(t => {
      const v = reliable && t.p ? t.p.medium : accuracyOf(t);
      return row(t.skill_name || t.topic, t.topic, v, false, reliable && t.p ? 'P(medium)' : 'accuracy');
    }).join('');
    const uRows = (unasked || []).map(u => row(u.skill_name || u.topic, u.topic, u.predicted_p, true, 'predicted')).join('');
    return `<div class="mvp-legend"><span class="legend-solid"></span> Measured (you answered questions on it)
        <span class="legend-hatched"></span> Predicted, not measured</div>
      <h4 class="mvp-h">Measured topics (${measured.length})</h4>${mRows || '<p class="small-note">None yet.</p>'}
      <h4 class="mvp-h">Inferred topics (${(unasked || []).length})</h4>${uRows || '<p class="small-note">All topics were measured.</p>'}
      <p class="small-note">Inferred topics were never asked. Their values come from the related-topic graph: the model links topics that students in the training data tended to practise together, and carries your answers on measured topics over to them. Probe questions check this.</p>`;
  }

  function predictionCheck(log) {
    const scored = log.filter(e => e.predicted_p !== null && e.predicted_p !== undefined && (e.correct === 0 || e.correct === 1));
    if (scored.length === 0) return { html: '<p class="small-note">No scored predictions in this session.</p>', scored };
    const n = scored.length;
    const brier = scored.reduce((s, e) => s + Math.pow(e.predicted_p - e.correct, 2), 0) / n;
    const match = scored.filter(e => (e.predicted_p >= 0.5 ? 1 : 0) === e.correct).length / n;
    const avgP = scored.reduce((s, e) => s + e.predicted_p, 0) / n;
    const avgA = scored.reduce((s, e) => s + e.correct, 0) / n;
    const html = `<div class="stat-row">
        ${statCard('Brier score', brier.toFixed(3), 'lower is better; 0.25 = always guessing 50%')}
        ${statCard('Prediction matched outcome', pct(match), 'p ≥ 50% vs. your result')}
        ${statCard('Average predicted', pct(avgP), `actual: ${pct(avgA)} correct`)}
      </div>
      <div class="chart-box"><canvas id="predVsActualChart" aria-label="Predicted probability versus actual outcome per question" role="img"></canvas></div>
      <p class="small-note caveat">Based on only ${n} question${n === 1 ? '' : 's'} - indicative, not statistically conclusive.</p>`;
    return { html, scored };
  }

  function drawPredChart(scored) {
    const canvas = $('predVsActualChart');
    if (!canvas || !window.Chart) return;
    if (state.predChart) state.predChart.destroy();
    state.predChart = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: scored.map(e => `Q${e.position}`),
        datasets: [
          { label: 'Predicted chance (before answering)', data: scored.map(e => Math.round(e.predicted_p * 100)), borderColor: '#2563eb', backgroundColor: '#2563eb', tension: 0.2, pointRadius: 4 },
          { label: 'Actual (100 = correct, 0 = not)', data: scored.map(e => e.correct * 100), borderColor: '#64748b', backgroundColor: scored.map(e => e.correct ? '#16a34a' : '#dc2626'), showLine: false, pointRadius: 6, pointStyle: 'rectRot' }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false, animation: REDUCED_MOTION ? false : undefined,
        scales: { y: { min: 0, max: 100, ticks: { callback: v => `${v}%` } } },
        plugins: { legend: { position: 'bottom' } } }
    });
  }

  function feedbackWidget() {
    const perceived = [['too_easy', 'Too easy'], ['about_right', 'About right'], ['too_hard', 'Too hard']];
    return `<div class="feedback-box" id="feedbackBox">
      <div class="fb-q">How did the difficulty feel? <span class="small-note">(optional)</span></div>
      <div class="fb-row" role="group" aria-label="Perceived difficulty">${perceived.map(([v, l]) => `<button type="button" class="fb-btn" data-perceived="${v}">${l}</button>`).join('')}</div>
      <div class="fb-q">Was this useful? (1 = not at all, 5 = very)</div>
      <div class="fb-row" role="group" aria-label="Helpfulness 1 to 5">${[1, 2, 3, 4, 5].map(v => `<button type="button" class="fb-btn" data-helpful="${v}">${v}</button>`).join('')}</div>
      <button type="button" class="submit-btn fb-send" id="feedbackSend" disabled>Send feedback</button>
      <div id="feedbackMsg" class="small-note" role="status"></div></div>`;
  }

  function wireFeedback(sessionId) {
    const box = $('feedbackBox');
    if (!box) return;
    const sel = { perceived: null, helpful: null };
    const send = $('feedbackSend');
    box.querySelectorAll('.fb-btn').forEach(btn => btn.addEventListener('click', () => {
      const key = btn.dataset.perceived ? 'perceived' : 'helpful';
      box.querySelectorAll(`.fb-btn[data-${key}]`).forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      btn.setAttribute('aria-pressed', 'true');
      sel[key] = key === 'helpful' ? Number(btn.dataset.helpful) : btn.dataset.perceived;
      send.disabled = !sel.perceived;
    }));
    send.addEventListener('click', async () => {
      send.disabled = true;
      const msg = $('feedbackMsg');
      try {
        const body = { perceived: sel.perceived };
        if (sel.helpful) body.helpful = sel.helpful;
        await fetchJSON(`/api/session/${encodeURIComponent(sessionId)}/feedback`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        msg.textContent = 'Thank you - feedback recorded.';
      } catch (e) {
        msg.textContent = 'Feedback could not be sent (optional feature unavailable).';
      }
    });
  }

  async function renderResultsV2(sessionId, studentId) {
    const root = $('resultsV2');
    if (!root) return false;
    let profile = null;
    try {
      profile = await fetchJSON(`/api/session/${encodeURIComponent(sessionId)}/profile`);
    } catch (e) {
      console.warn('[GKT] profile unavailable, keeping existing mastery view', e);
    }
    try {
      const log = state.sessionLog;
      const answered = log.filter(e => e.correct === 0 || e.correct === 1);
      const correct = answered.filter(e => e.correct === 1).length;
      let html = '';

      // Summary stat cards
      let strongest = '—', weakest = '—';
      if (profile && Array.isArray(profile.topics) && profile.topics.length) {
        const order = ['needs_support', 'developing', 'not_enough_evidence', 'proficient', 'mastered'];
        const score = t => (order.indexOf(t.status) * 10) + ((profile.profile_reliable && t.p) ? t.p.medium : (accuracyOf(t) || 0));
        const sorted = profile.topics.slice().sort((a, b) => score(b) - score(a));
        strongest = sorted[0].skill_name; weakest = sorted[sorted.length - 1].skill_name;
      }
      html += `<div class="stat-row">${statCard('Questions answered', answered.length)}${statCard('Correct answers', `${correct} of ${answered.length}`)}${statCard('Strongest topic', strongest)}${statCard('Weakest topic', weakest)}</div>`;

      if (profile && Array.isArray(profile.topics)) {
        const strong = profile.topics.filter(t => t.status === 'proficient' || t.status === 'mastered');
        const weak = profile.topics.filter(t => t.status === 'needs_support' || t.status === 'developing')
          .sort((a, b) => (a.status === 'needs_support' ? 0 : 1) - (b.status === 'needs_support' ? 0 : 1));
        const nee = profile.topics.filter(t => t.status === 'not_enough_evidence');
        const th = profile.thresholds || {};
        html += `<div class="results-boxes">
          <section class="result-box box-good" aria-labelledby="hStrong"><h3 id="hStrong"><span aria-hidden="true">✓</span> Strong topics</h3>${strong.map(topicCard).join('') || '<p class="small-note">No topic reached Proficient yet.</p>'}</section>
          <section class="result-box box-bad" aria-labelledby="hWeak"><h3 id="hWeak"><span aria-hidden="true">!</span> Needs improvement</h3>${weak.map(topicCard).join('') || '<p class="small-note">No topic flagged.</p>'}</section>
        </div>
        ${nee.length ? `<section class="result-box box-neutral" aria-labelledby="hNee"><h3 id="hNee"><span aria-hidden="true">?</span> Not enough evidence yet</h3>${nee.map(topicCard).join('')}</section>` : ''}
        <p class="small-note">Status labels are rule-based estimates from the model and your answers. ${th.T_PROF !== undefined ? `Proficient at a level = predicted success ≥ ${pct(th.T_PROF)}; a status needs at least ${th.MIN_ANSWERS} answers on the topic, and Mastered needs a correct Hard answer.` : ''}</p>`;

        const measured = profile.topics.filter(t => t.n_answers > 0);
        html += `<section class="v2-section"><h3>Difficulty profile per topic</h3>${difficultyProfile(measured, !!profile.profile_reliable)}</section>`;
        html += `<section class="v2-section"><h3>Measured vs. predicted</h3>${measuredVsPredicted(profile.topics, profile.unasked, !!profile.profile_reliable)}</section>`;
      } else {
        html += '<p class="small-note">Detailed topic profile unavailable; showing the standard mastery estimates below.</p>';
      }

      const pc = predictionCheck(log);
      html += `<section class="v2-section"><h3>Did the prediction work?</h3>${pc.html}</section>`;
      html += `<section class="v2-section"><h3>Quick feedback</h3>${feedbackWidget()}</section>`;
      root.innerHTML = html;
      drawPredChart(pc.scored);
      wireFeedback(sessionId);
      return !!profile;
    } catch (e) {
      console.warn('[GKT] renderResultsV2 failed', e);
      return false;
    }
  }

  // ------------------------------------------------------------------
  window.GKT = {
    BAND_HIGH, BAND_MID, SHOW_VERDICT, SHOW_ADAPTATION_TOAST,
    band, statusMeta, levelColors, esc, state,
    resetSession: safe('resetSession', resetSession),
    onQuestion: safe('onQuestion', onQuestion),
    onSubmitStart: safe('onSubmitStart', onSubmitStart),
    onAnswer: safe('onAnswer', onAnswer),
    onSubmitError: safe('onSubmitError', onSubmitError),
    renderResultsV2
  };
})();
