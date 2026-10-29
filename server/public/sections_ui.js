/*
 * sections_ui.js (v3)
 * -------------------
 * Additive, feature-detected UI for learner-level sections:
 *  - a "Learner level" picker on the login card (only if GET /api/sections works;
 *    default option = the original topics, so nothing changes if it is not used),
 *  - the section name and its validation label ("validated on real data" /
 *    "transfer mode, not validated") on the quiz header, the results page and
 *    the teacher dashboard,
 *  - answer choices for generated multiple-choice items (clicking a choice only
 *    fills the answer box; no correct/incorrect verdict is ever shown),
 *  - an input hint for fraction answers.
 * Loaded after adaptive_ui.js and before app.js / dashboard.js. Every entry point
 * is wrapped in try/catch so a failure here never breaks the quiz.
 */
(function () {
  'use strict';

  var state = { sections: null, current: null };

  function el(tag, attrs, text) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  function validationText(s) {
    return s && s.validated ? 'validated on real data' : 'transfer mode, not validated';
  }

  function sectionById(id) {
    if (!state.sections) return null;
    for (var i = 0; i < state.sections.length; i++) {
      if (state.sections[i].section === id) return state.sections[i];
    }
    return null;
  }

  // ---- login card picker ----------------------------------------------------
  function renderPicker() {
    var form = document.getElementById('loginForm');
    if (!form || document.getElementById('sectionSelect')) return;
    var wrap = el('div', { 'class': 'section-picker', id: 'sectionPicker' });
    var lab = el('label', { 'for': 'sectionSelect', 'class': 'section-label' }, 'Learner level');
    var sel = el('select', { id: 'sectionSelect', 'class': 'section-select' });
    sel.appendChild(el('option', { value: '' }, 'Default - original topics (validated on real data)'));
    state.sections.forEach(function (s) {
      sel.appendChild(el('option', { value: s.section }, s.label + ' - ' + s.grade_label + ' (' + validationText(s) + ')'));
    });
    var note = el('p', { id: 'sectionNote', 'class': 'section-note', 'aria-live': 'polite' });
    sel.addEventListener('change', function () { updateNote(sel.value, note); });
    wrap.appendChild(lab);
    wrap.appendChild(sel);
    wrap.appendChild(note);
    form.appendChild(wrap);
    updateNote('', note);
  }

  function updateNote(value, note) {
    var s = sectionById(value);
    if (!s) {
      note.textContent = 'Original 10 ASSISTments topics.';
      note.className = 'section-note';
      return;
    }
    var topics = (s.topics || []).map(function (t) { return t.name; }).join(', ');
    note.textContent = topics + '. ' + s.validation_note;
    note.className = 'section-note ' + (s.validated ? 'is-validated' : 'is-transfer');
  }

  // ---- labels on quiz header and results page -------------------------------
  function badgeText(d) {
    var label = d && d.section_label ? d.section_label : 'Default topics';
    return label + ' · ' + validationText(d);
  }

  function renderBadges() {
    var d = state.current;
    if (!d) return;
    var cls = 'section-badge ' + (d.validated ? 'is-validated' : 'is-transfer');
    var anchor = document.getElementById('activeStudentBadge');
    if (anchor && anchor.parentNode) {
      var b = document.getElementById('sectionBadge');
      if (!b) {
        b = el('span', { id: 'sectionBadge' });
        anchor.parentNode.insertBefore(b, anchor.nextSibling);
      }
      b.className = cls;
      b.textContent = badgeText(d);
    }
    var results = document.getElementById('resultsSection');
    if (results) {
      var r = document.getElementById('sectionResultsBanner');
      if (!r) {
        r = el('div', { id: 'sectionResultsBanner', role: 'note' });
        var title = results.querySelector('h2');
        if (title && title.parentNode === results) results.insertBefore(r, title.nextSibling);
        else results.insertBefore(r, results.firstChild);
      }
      r.className = cls + ' section-results';
      var noModel = d.model_predictions === false || (d.model_predictions === undefined && !d.validated);
      r.textContent = 'Section: ' + badgeText(d) +
        (noModel ? '. The served model was not trained on this level, so no model predictions are shown; results use your answers only.' : '');
    }
  }

  // ---- answer choices / input hints ------------------------------------------
  function renderChoices(q) {
    var form = document.getElementById('answerForm');
    var input = document.getElementById('answerInput');
    if (!form || !input) return;
    var box = document.getElementById('answerChoices');
    if (!box) {
      box = el('div', { id: 'answerChoices', 'class': 'answer-choices', role: 'group', 'aria-label': 'Answer choices' });
      form.appendChild(box);
    }
    box.innerHTML = '';
    if (!input.getAttribute('data-default-placeholder')) {
      input.setAttribute('data-default-placeholder', input.getAttribute('placeholder') || '');
    }
    input.setAttribute('placeholder', input.getAttribute('data-default-placeholder'));
    if (q && q.answer_type === 'fraction') input.setAttribute('placeholder', 'Type a fraction, e.g. 3/4');
    if (!q || q.answer_type !== 'mc' || !Array.isArray(q.choices)) {
      box.style.display = 'none';
      return;
    }
    box.style.display = '';
    q.choices.forEach(function (c, i) {
      var btn = el('button', { type: 'button', 'class': 'choice-btn' }, String.fromCharCode(65 + i) + ') ' + c);
      btn.addEventListener('click', function () {
        input.value = c;
        Array.prototype.forEach.call(box.children, function (x) { x.classList.remove('is-selected'); });
        btn.classList.add('is-selected');
        input.focus();
      });
      box.appendChild(btn);
    });
  }

  // In evidence-only sections there is never a model prediction, so the card must
  // not say "warm-up" after question 2.
  function relabelPrediction(q) {
    var d = state.current;
    if (!d || d.model_predictions !== false || !q || q.predicted_p !== null) return;
    var label = document.getElementById('predCardLabel');
    var sub = document.getElementById('predCardSubtitle');
    if (label) label.textContent = 'No model prediction in this section';
    if (sub) sub.textContent = 'The served model was not trained on this level; topics follow your answers';
  }

  // ---- teacher dashboard: one view per learner level ----------------------------
  // Tabs Default / Primary / Secondary / Undergraduate. The chosen level is kept in the
  // URL (dashboard.html?section=primary) so each level has its own link. A level view
  // shows only that level's topics and students in the class overview; the original
  // ASSISTments mastery matrix is hidden there because it only covers the original topics.
  state.dashSection = null;

  function legacyBlocks() {
    var out = [];
    var matrix = document.getElementById('matrixTitle');
    if (matrix && matrix.closest('.card')) out.push(matrix.closest('.card'));
    var stat = document.getElementById('statStudentCount');
    if (stat && stat.closest('.card') && stat.closest('.card').parentElement) out.push(stat.closest('.card').parentElement);
    var tabs = document.querySelector('.tab-group');
    if (tabs) out.push(tabs);
    var empty = document.getElementById('emptyState');
    if (empty) out.push(empty);
    return out;
  }

  function selectDashSection(id, push) {
    state.dashSection = id || null;
    var tabs = document.querySelectorAll('#sectionDashTabs button');
    Array.prototype.forEach.call(tabs, function (b) {
      var on = (b.getAttribute('data-section') || '') === (id || '');
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', String(on));
    });
    var info = document.getElementById('sectionDashInfo');
    var s = sectionById(id);
    if (info) {
      info.innerHTML = '';
      if (!s) {
        info.appendChild(el('span', { 'class': 'section-badge is-validated' }, 'validated on real data'));
        info.appendChild(el('span', { 'class': 'section-topics' },
          ' Original 10 ASSISTments topics (Grades 6-10); model predictions shown.'));
      } else {
        info.appendChild(el('span', { 'class': 'section-badge ' + (s.validated ? 'is-validated' : 'is-transfer') }, validationText(s)));
        info.appendChild(el('span', { 'class': 'section-topics' }, ' ' + s.grade_label + '. Topics: ' +
          (s.topics || []).map(function (t) { return t.name; }).join(', ') + '. ' +
          (s.serving === 'model' ? 'Model predictions shown.' : 'Evidence-only: statuses come from the students\' answers; no model predictions.')));
        info.appendChild(el('p', { 'class': 'section-note' }, s.validation_note));
      }
    }
    legacyBlocks().forEach(function (n) {
      if (n.id === 'emptyState') { if (id) n.style.display = 'none'; return; }
      n.style.display = id ? 'none' : '';
    });
    var hidden = document.getElementById('sectionLegacyNote');
    if (hidden) hidden.style.display = id ? '' : 'none';
    if (push) {
      try {
        var url = new URL(window.location.href);
        if (id) url.searchParams.set('section', id); else url.searchParams.delete('section');
        window.history.replaceState(null, '', url.toString());
      } catch (e) { /* ignore */ }
    }
    if (typeof window.reloadClassOverview === 'function') window.reloadClassOverview();
    var card = document.getElementById('classOverviewCard');
    if (push && id && card && card.scrollIntoView) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderDashboardCard() {
    if (!/dashboard/.test(window.location.pathname) || document.getElementById('sectionsInfoCard')) return;
    var main = document.querySelector('main') || document.querySelector('.container') || document.body;
    var card = el('section', { id: 'sectionsInfoCard', 'class': 'sections-info-card', 'aria-label': 'Dashboard by learner level' });
    card.appendChild(el('h3', {}, 'Dashboard by learner level'));
    var tabs = el('div', { id: 'sectionDashTabs', 'class': 'section-dash-tabs', role: 'group', 'aria-label': 'Choose a learner level' });
    var levels = [{ section: '', label: 'Default (original topics)' }].concat(state.sections);
    levels.forEach(function (s) {
      var b = el('button', { type: 'button', 'data-section': s.section || '', 'aria-pressed': 'false' }, s.label);
      b.addEventListener('click', function () { selectDashSection(s.section || null, true); });
      tabs.appendChild(b);
    });
    card.appendChild(tabs);
    card.appendChild(el('div', { id: 'sectionDashInfo', 'class': 'section-dash-info', 'aria-live': 'polite' }));
    var note = el('p', { id: 'sectionLegacyNote', 'class': 'section-note' },
      'The ASSISTments mastery matrix is hidden for this level (it only covers the original topics). The class overview below shows this level\'s students and topics.');
    note.style.display = 'none';
    card.appendChild(note);
    main.insertBefore(card, main.firstChild);
    var initial = null;
    try { initial = new URL(window.location.href).searchParams.get('section'); } catch (e) { initial = null; }
    selectDashSection(sectionById(initial) ? initial : null, false);
  }

  window.getDashboardSection = function () { return state.dashSection; };

  // After each class-overview load: in evidence-only levels the footer must not say
  // that statuses come from the model.
  var defaultOverviewNote = null;
  window.gktOverviewLoaded = function (data) {
    var n = document.getElementById('classOverviewNote');
    if (!n) return;
    if (defaultOverviewNote === null) defaultOverviewNote = n.textContent;
    n.textContent = (data && data.model_predictions === false)
      ? 'Statuses are rule-based estimates from the students\' answers only (no model predictions for this level); suggestions are rule-based, for the teacher to judge.'
      : defaultOverviewNote;
  };

  async function loadSections() {
    try {
      var r = await fetch('/api/sections');
      if (!r.ok) return;
      var d = await r.json();
      if (!d || !Array.isArray(d.sections) || !d.sections.length) return;
      state.sections = d.sections;
      renderPicker();
      renderDashboardCard();
    } catch (e) {
      /* sections unavailable: keep the original UI */
    }
  }

  // ---- public hooks (used by app.js through feature detection) ----------------
  window.getSelectedSection = function () {
    try {
      var s = document.getElementById('sectionSelect');
      return s && s.value ? s.value : null;
    } catch (e) { return null; }
  };
  window.gktSectionStarted = function (data) {
    try {
      state.current = data && Object.prototype.hasOwnProperty.call(data, 'validated') ? data : null;
      var old = document.getElementById('sectionBadge');
      if (!state.current && old) old.remove();
      var oldR = document.getElementById('sectionResultsBanner');
      if (!state.current && oldR) oldR.remove();
      renderBadges();
    } catch (e) { /* ignore */ }
  };

  // Wrap the adaptive UI's question hook so generated items get choices / hints.
  if (window.GKT && typeof window.GKT.onQuestion === 'function') {
    var orig = window.GKT.onQuestion;
    window.GKT.onQuestion = function (q) {
      var out = orig.apply(this, arguments);
      try { renderChoices(q); renderBadges(); relabelPrediction(q); } catch (e) { /* ignore */ }
      return out;
    };
  } else {
    window.GKT = window.GKT || {};
    window.GKT.onQuestion = function (q) { try { renderChoices(q); } catch (e) { /* ignore */ } };
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', loadSections);
  else loadSections();
})();
