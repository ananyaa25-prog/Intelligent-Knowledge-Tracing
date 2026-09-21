/**
 * server/public/dashboard.js
 * --------------------------
 * Frontend controller for Part B Class-Level Teacher Dashboard.
 *
 * NOTE: A demo dashboard for visualizing model output across multiple students,
 * built for presentation purposes -- not a research contribution.
 */

window.addEventListener('DOMContentLoaded', () => {
  loadDashboardData();
  document.getElementById('refreshBtn').addEventListener('click', loadDashboardData);
});

async function loadDashboardData() {
  const loadingIndicator = document.getElementById('loadingIndicator');
  const heatmapWrapper = document.getElementById('heatmapWrapper');
  const thead = document.getElementById('heatmapThead');
  const tbody = document.getElementById('heatmapTbody');

  loadingIndicator.style.display = 'block';
  heatmapWrapper.style.display = 'none';

  try {
    const res = await fetch('/api/dashboard/matrix');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const { students, skills } = data;

    // Update Summary Statistics
    document.getElementById('statStudentCount').textContent = students.length;
    document.getElementById('statSkillCount').textContent = skills.length;

    let totalInteractions = 0;
    let allMasteryVals = [];

    students.forEach(s => {
      totalInteractions += s.num_interactions;
      Object.values(s.mastery).forEach(v => allMasteryVals.push(v));
    });

    document.getElementById('statInteractionCount').textContent = totalInteractions.toLocaleString();
    const overallAvg = allMasteryVals.length > 0
      ? (allMasteryVals.reduce((a, b) => a + b, 0) / allMasteryVals.length * 100).toFixed(1) + '%'
      : '--';
    document.getElementById('statAvgMastery').textContent = overallAvg;

    // Build Table Header
    thead.innerHTML = '';
    const headerRow = document.createElement('tr');
    
    // Top-left student column header
    const thStudent = document.createElement('th');
    thStudent.className = 'sticky-col';
    thStudent.textContent = 'Student ID';
    headerRow.appendChild(thStudent);

    // Skill columns
    skills.forEach(sk => {
      const th = document.createElement('th');
      th.title = `${sk.name} (ID: ${sk.skill_id})`;
      th.innerHTML = `
        <div style="font-size: 0.7rem; color: var(--text-muted);">#${sk.skill_id}</div>
        <div style="max-width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${sk.name}
        </div>
      `;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    // Build Table Body (Rows = Students)
    tbody.innerHTML = '';
    students.forEach(st => {
      const tr = document.createElement('tr');

      // Student name column
      const tdStudent = document.createElement('td');
      tdStudent.className = 'sticky-col';
      tdStudent.innerHTML = `
        <div>${st.student_id}</div>
        <div style="font-size: 0.68rem; color: var(--text-muted); font-weight: normal;">
          ${st.num_interactions} interactions
        </div>
      `;
      tr.appendChild(tdStudent);

      // Skill cells
      skills.forEach(sk => {
        const td = document.createElement('td');
        const masteryVal = st.mastery[sk.skill_id];

        if (masteryVal !== undefined && masteryVal !== null) {
          const pct = (masteryVal * 100).toFixed(0);
          td.textContent = `${pct}%`;
          td.title = `${st.student_id} on ${sk.name} (ID: ${sk.skill_id})\nMastery: ${(masteryVal * 100).toFixed(1)}% (${masteryVal.toFixed(4)})`;

          if (masteryVal >= 0.75) {
            td.className = 'cell-high';
          } else if (masteryVal >= 0.50) {
            td.className = 'cell-med';
          } else {
            td.className = 'cell-low';
          }
        } else {
          td.textContent = '—';
          td.className = 'empty-cell';
          td.title = `${st.student_id} has no recorded interactions for ${sk.name}`;
        }

        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });

    // Add Class Average Footer Row
    const avgRow = document.createElement('tr');
    avgRow.style.fontWeight = 'bold';
    avgRow.style.background = '#f1f5f9';

    const tdAvgLabel = document.createElement('td');
    tdAvgLabel.className = 'sticky-col';
    tdAvgLabel.textContent = 'Class Average';
    avgRow.appendChild(tdAvgLabel);

    skills.forEach(sk => {
      const td = document.createElement('td');
      const valsForSkill = [];
      students.forEach(st => {
        if (st.mastery[sk.skill_id] !== undefined) {
          valsForSkill.push(st.mastery[sk.skill_id]);
        }
      });

      if (valsForSkill.length > 0) {
        const skillAvg = valsForSkill.reduce((a, b) => a + b, 0) / valsForSkill.length;
        td.textContent = `${(skillAvg * 100).toFixed(0)}%`;
        td.title = `Class Average for ${sk.name}: ${(skillAvg * 100).toFixed(1)}% across ${valsForSkill.length} students`;
        if (skillAvg >= 0.75) td.className = 'cell-high';
        else if (skillAvg >= 0.50) td.className = 'cell-med';
        else td.className = 'cell-low';
      } else {
        td.textContent = '—';
        td.className = 'empty-cell';
      }
      avgRow.appendChild(td);
    });
    tbody.appendChild(avgRow);

    loadingIndicator.style.display = 'none';
    heatmapWrapper.style.display = 'block';
  } catch (err) {
    loadingIndicator.innerHTML = `
      <p style="color: var(--accent-rose); font-weight: 700;">Failed to load dashboard data</p>
      <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">${err.message}</p>
    `;
  }
}
