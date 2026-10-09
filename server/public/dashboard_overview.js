/**
 * server/public/dashboard_overview.js
 * -----------------------------------
 * adaptive-v2 (T8a): class overview card on the teacher dashboard.
 * Students x topics heatmap coloured by profile STATUS (icon + text + colour),
 * evidence counts, a "Needs attention" list, and a Live / Simulated demo toggle.
 * Data: GET /api/class/overview?source=live|sample. Live and simulated data are
 * never mixed; simulated data is always labelled. Degrades to an explanatory
 * empty state if the endpoint is missing.
 */
(function () {
  'use strict';
  const STATUS_ORDER = ['needs_support', 'developing', 'proficient', 'mastered', 'not_enough_evidence'];
  const SHORT = { needs_support: 'Support', developing: 'Dev.', proficient: 'Prof.', mastered: 'Mast.', not_enough_evidence: '?' };
  let source = 'live';

  const $ = id => document.getElementById(id);
  const esc = s => (window.GKT ? window.GKT.esc(s) : String(s));
  const meta = s => (window.GKT ? window.GKT.statusMeta(s) : { label: s, icon: '', bg: '#f1f5f9', text: '#334155', fg: '#64748b' });

  function legend() {
    return STATUS_ORDER.map(s => {
      const m = meta(s);
      return `<span class="status-chip" style="background:${m.bg};color:${m.text};border-color:${m.fg}"><span aria-hidden="true">${m.icon}</span> ${esc(m.label)}</span>`;
    }).join('') + '<span class="evidence-chip">number = answers on that topic</span><span class="evidence-chip">blank = not assessed</span>';
  }

  function render(data) {
    const body = $('classOverviewBody');
    const empty = $('classOverviewEmpty');
    const label = $('classOverviewLabel');
    label.innerHTML = source === 'sample'
      ? `<span class="sim-label">Simulated demo students</span> <span class="small-note">${esc(data.label || '')}</span>`
      : `<span class="small-note">${esc(data.label || 'Live students')}</span>`;
    const students = data.students || [];
    if (!students.length) {
      body.style.display = 'none';
      empty.style.display = 'block';
      empty.textContent = source === 'sample'
        ? 'No simulated demo students in the database. Seed them with: python scripts/seed_demo_data.py --yes-real-db'
        : 'No live students with answers yet (test, smoke and simulated accounts are excluded). Complete a quiz on the Live Quiz page.';
      return;
    }
    empty.style.display = 'none';
    body.style.display = 'block';
    const skills = data.skills || [];
    const head = `<tr><th class="st-name">Student</th>${skills.map(s => `<th title="${esc(s.name)} (ID ${esc(s.skill_id)})">${esc(s.name.length > 14 ? s.name.slice(0, 13) + '…' : s.name)}</th>`).join('')}</tr>`;
    const rows = students.map(st => `<tr><th class="st-name" scope="row">${esc(st.student_id)}</th>${skills.map(sk => {
      const c = (st.topics || {})[sk.skill_id];
      if (!c) return '<td style="background:#fafafa;color:#94a3b8" title="Not assessed">·</td>';
      const m = meta(c.status);
      return `<td style="background:${m.bg};color:${m.text};box-shadow: inset 0 0 0 1.5px ${m.fg}" title="${esc(st.student_id)} - ${esc(sk.name)}: ${esc(m.label)} (${c.n_answers} answers)"><span aria-hidden="true">${m.icon}</span> ${c.n_answers}<span class="sr-only"> ${esc(m.label)}</span></td>`;
    }).join('')}</tr>`).join('');
    $('classOverviewHeatmap').innerHTML = `<thead>${head}</thead><tbody>${rows}</tbody>`;
    $('classOverviewLegend').innerHTML = legend();
    const counts = data.status_counts || {};
    $('classOverviewCounts').textContent = `${students.length} student(s). Topic statuses: ` +
      STATUS_ORDER.filter(s => counts[s]).map(s => `${meta(s).label} ${counts[s]}`).join(', ');
    const att = data.needs_attention || [];
    $('needsAttentionList').innerHTML = att.map(a => `<li><strong>${esc(a.student_id)}</strong> - ${esc(a.skill_name)}: ${esc(meta(a.status).label)} (${a.n_answers} answers). Suggestion: ${esc(a.recommended_action)}</li>`).join('');
    renderAttention(data);
  }

  // ---- Needs attention: grouped by student and by topic -------------------
  let naMode = 'student';
  let naShowAll = false;
  let lastData = null;
  const NA_LIMIT = 6;

  function chip(status, text, title) {
    const m = meta(status);
    return `<span class="na-chip" style="background:${m.bg};color:${m.text};border-color:${m.fg}" title="${esc(title || '')}"><span aria-hidden="true">${m.icon}</span> ${esc(text)}</span>`;
  }

  function renderAttention(data) {
    const names = {};
    (data.skills || []).forEach(s => { names[s.skill_id] = s.name; });
    const students = data.students || [];
    const byStudent = students.map(st => {
      const support = [];
      const developing = [];
      Object.entries(st.topics || {}).forEach(([tid, c]) => {
        const item = { tid, name: names[tid] || `Topic ${tid}`, n: c.n_answers };
        if (c.status === 'needs_support') support.push(item);
        else if (c.status === 'developing') developing.push(item);
      });
      support.sort((x, y) => y.n - x.n);
      return { id: st.student_id, support, developing };
    }).filter(r => r.support.length > 0)
      .sort((x, y) => y.support.length - x.support.length || x.id.localeCompare(y.id));

    const byTopic = {};
    students.forEach(st => Object.entries(st.topics || {}).forEach(([tid, c]) => {
      const t = byTopic[tid] || (byTopic[tid] = { tid, name: names[tid] || `Topic ${tid}`, support: [], developing: 0, assessed: 0 });
      if (c.status !== 'not_enough_evidence') t.assessed += 1;
      if (c.status === 'needs_support') t.support.push(st.student_id);
      if (c.status === 'developing') t.developing += 1;
    }));
    const topics = Object.values(byTopic).filter(t => t.support.length > 0)
      .sort((x, y) => y.support.length - x.support.length);

    const top = topics[0];
    $('naSummary').innerHTML = byStudent.length
      ? `<span class="na-stat"><strong>${byStudent.length}</strong> of ${students.length} student(s) need support on at least one topic</span>`
        + (top ? `<span class="na-stat">Most common: <strong>${esc(top.name)}</strong> (${top.support.length} student${top.support.length === 1 ? '' : 's'})</span>` : '')
        + '<span class="na-stat muted">A topic gets a status only after 2 answers</span>'
      : '<span class="na-stat ok">No student-topic pair is currently rated Needs support.</span>';

    const list = naShowAll ? byStudent : byStudent.slice(0, NA_LIMIT);
    $('naByStudent').innerHTML = list.map(r => `
      <div class="na-card">
        <div class="na-card-head"><strong>${esc(r.id)}</strong>
          <span class="na-count">${r.support.length} topic${r.support.length === 1 ? '' : 's'}</span></div>
        <div class="na-chips">${r.support.slice(0, 4).map(t => chip('needs_support', `${t.name} (${t.n} ans)`, `${t.name}: ${t.n} answers`)).join('')}${r.support.length > 4 ? `<span class="na-chip na-more" title="${esc(r.support.slice(4).map(t => t.name).join(', '))}">+${r.support.length - 4} more</span>` : ''}</div>
        ${r.developing.length ? `<div class="na-sub">Also developing: ${r.developing.map(t => esc(t.name)).join(', ')}</div>` : ''}
        <div class="na-action">${r.support.length >= 5
          ? 'Suggested (rule-based): broad review - most assessed topics need support; start with Easy practice on <strong>' + esc(r.support[0].name) + '</strong>.'
          : 'Suggested (rule-based): Easy practice on <strong>' + esc(r.support[0].name) + '</strong>' + (r.support.length > 1 ? ' first, then the other ' + (r.support.length - 1) + '.' : '.')}</div>
      </div>`).join('');

    $('naByTopic').innerHTML = topics.length ? `<div class="overview-scroll"><table class="na-table"><thead><tr><th>Topic</th><th>Needs support</th><th>Developing</th><th>Suggested (rule-based)</th></tr></thead><tbody>${
      topics.map(t => {
        const share = t.assessed ? t.support.length / t.assessed : 0;
        const action = (t.support.length >= 3 || share >= 0.3) ? 'Re-teach to the group' : 'Individual Easy practice';
        return `<tr><td><strong>${esc(t.name)}</strong><div class="na-sub">${t.assessed} student(s) with a status</div></td>
          <td><span class="na-count">${t.support.length}</span><div class="na-sub">${t.support.map(id => esc(id)).join(', ')}</div></td>
          <td>${t.developing}</td><td>${action}</td></tr>`;
      }).join('')}</tbody></table></div>` : '';

    const btn = $('naShowAll');
    if (naMode === 'student' && byStudent.length > NA_LIMIT) {
      btn.style.display = 'inline-block';
      btn.textContent = naShowAll ? 'Show fewer' : `Show all ${byStudent.length} students`;
    } else {
      btn.style.display = 'none';
    }
    $('naByStudent').style.display = naMode === 'student' ? 'grid' : 'none';
    $('naByTopic').style.display = naMode === 'topic' ? 'block' : 'none';
  }

  function setNaMode(m) {
    naMode = m;
    $('naByStudentBtn').classList.toggle('active', m === 'student');
    $('naByTopicBtn').classList.toggle('active', m === 'topic');
    $('naByStudentBtn').setAttribute('aria-pressed', String(m === 'student'));
    $('naByTopicBtn').setAttribute('aria-pressed', String(m === 'topic'));
    if (lastData) renderAttention(lastData);
  }

  async function load() {
    const empty = $('classOverviewEmpty');
    try {
      const r = await fetch(`/api/class/overview?source=${encodeURIComponent(source)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      lastData = await r.json();
      render(lastData);
    } catch (e) {
      $('classOverviewBody').style.display = 'none';
      empty.style.display = 'block';
      empty.textContent = 'Class overview is unavailable (the backend endpoint did not respond). The mastery matrix above still works.';
      console.warn('class overview unavailable', e);
    }
  }

  function setSource(s) {
    source = s;
    $('overviewLiveBtn').classList.toggle('active', s === 'live');
    $('overviewSimBtn').classList.toggle('active', s === 'sample');
    $('overviewLiveBtn').setAttribute('aria-pressed', String(s === 'live'));
    $('overviewSimBtn').setAttribute('aria-pressed', String(s === 'sample'));
    load();
  }

  window.addEventListener('DOMContentLoaded', () => {
    try {
      if (!$('classOverviewCard')) return;
      $('overviewLiveBtn').addEventListener('click', () => setSource('live'));
      $('overviewSimBtn').addEventListener('click', () => setSource('sample'));
      $('naByStudentBtn').addEventListener('click', () => setNaMode('student'));
      $('naByTopicBtn').addEventListener('click', () => setNaMode('topic'));
      $('naShowAll').addEventListener('click', () => { naShowAll = !naShowAll; if (lastData) renderAttention(lastData); });
      const refresh = $('refreshBtn');
      if (refresh) refresh.addEventListener('click', load);
      load();
    } catch (e) {
      console.warn('class overview init failed', e);
    }
  });
})();
