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
    $('needsAttentionList').innerHTML = att.length
      ? att.slice(0, 20).map(a => `<li><strong>${esc(a.student_id)}</strong> - ${esc(a.skill_name)}: ${esc(meta(a.status).label)} (${a.n_answers} answers). Suggestion: ${esc(a.recommended_action)}</li>`).join('')
      : '<li style="border-left-color:#16a34a">No student-topic pair is currently flagged as Needs support.</li>';
  }

  async function load() {
    const empty = $('classOverviewEmpty');
    try {
      const r = await fetch(`/api/class/overview?source=${encodeURIComponent(source)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      render(await r.json());
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
      const refresh = $('refreshBtn');
      if (refresh) refresh.addEventListener('click', load);
      load();
    } catch (e) {
      console.warn('class overview init failed', e);
    }
  });
})();
