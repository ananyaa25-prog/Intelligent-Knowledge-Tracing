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

  // ---- teacher dashboard: sections card ---------------------------------------
  function renderDashboardCard() {
    if (!/dashboard/.test(window.location.pathname) || document.getElementById('sectionsInfoCard')) return;
    var main = document.querySelector('main') || document.querySelector('.container') || document.body;
    var card = el('section', { id: 'sectionsInfoCard', 'class': 'sections-info-card', 'aria-label': 'Learner-level sections' });
    card.appendChild(el('h3', {}, 'Learner-level sections'));
    var list = el('ul', { 'class': 'sections-info-list' });
    state.sections.forEach(function (s) {
      var li = el('li', {});
      li.appendChild(el('strong', {}, s.label + ' (' + s.grade_label + '): '));
      li.appendChild(el('span', { 'class': 'section-badge ' + (s.validated ? 'is-validated' : 'is-transfer') }, validationText(s)));
      li.appendChild(el('span', { 'class': 'section-topics' }, ' ' + (s.topics || []).map(function (t) { return t.name; }).join(', ')));
      list.appendChild(li);
    });
    card.appendChild(list);
    card.appendChild(el('p', { 'class': 'section-note' },
      'The class overview below uses the original topics. Section sessions are labelled on each student\'s results page.'));
    main.insertBefore(card, main.firstChild);
  }

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
