/**
 * server/public/dashboard.js
 * --------------------------
 * Frontend controller for Part B & Task 7:
 * Class-Level Teacher Dashboard querying live SQLite database with sample historical toggle.
 *
 * NOTE: A demo dashboard for visualizing model output across multiple students,
 * built for presentation purposes -- not a research contribution.
 */

let currentSource = 'live';

window.addEventListener('DOMContentLoaded', () => {
  const tabLive = document.getElementById('tabLive');
  const tabSample = document.getElementById('tabSample');
  const refreshBtn = document.getElementById('refreshBtn');

  if (tabLive) {
    tabLive.addEventListener('click', () => {
      currentSource = 'live';
      tabLive.classList.add('active');
      tabSample.classList.remove('active');
      document.getElementById('dashboardSubtitle').textContent =
        'Visualizes cumulative per-skill mastery predictions across registered live students.';
      loadDashboardData();
    });
  }

  if (tabSample) {
    tabSample.addEventListener('click', () => {
      currentSource = 'sample';
      tabSample.classList.add('active');
      tabLive.classList.remove('active');
      document.getElementById('dashboardSubtitle').textContent =
        'Visualizes per-skill mastery predictions across historical ASSISTments test-split students.';
      loadDashboardData();
    });
  }

  if (refreshBtn) {
    refreshBtn.addEventListener('click', loadDashboardData);
  }

  loadDashboardData();
});

async function loadDashboardData() {
  const loadingIndicator = document.getElementById('loadingIndicator');
  const emptyState = document.getElementById('emptyState');
  const heatmapWrapper = document.getElementById('heatmapWrapper');
  const thead = document.getElementById('heatmapThead');
  const tbody = document.getElementById('heatmapTbody');

  loadingIndicator.style.display = 'block';
  emptyState.style.display = 'none';
  heatmapWrapper.style.display = 'none';

  try {
    const res = await fetch(`/api/dashboard/matrix?source=${encodeURIComponent(currentSource)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const { students, skills } = data;

    // Handle Empty State for Live Data
    if (!students || students.length === 0) {
      loadingIndicator.style.display = 'none';
      emptyState.style.display = 'block';
      heatmapWrapper.style.display = 'none';

      document.getElementById('statStudentCount').textContent = '0';
      document.getElementById('statSkillCount').textContent = (skills && skills.length) || 10;
      document.getElementById('statInteractionCount').textContent = '0';
      document.getElementById('statAvgMastery').textContent = '--';
      return;
    }

    loadingIndicator.style.display = 'none';
    emptyState.style.display = 'none';
    heatmapWrapper.style.display = 'block';

    // Update Summary Statistics
    document.getElementById('statStudentCount').textContent = students.length;
    document.getElementById('statSkillCount').textContent = skills.length;

    let totalInteractions = 0;
    let allMasteryVals = [];

    students.forEach(s => {
      totalInteractions += s.num_interactions || 0;
      Object.values(s.mastery || {}).forEach(v => {
        if (typeof v === 'number' && !isNaN(v)) {
          allMasteryVals.push(v);
        }
      });
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
        <div style="max-width: 95px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
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
      tdStudent.style.cursor = 'pointer';
      const sessInfo = st.completed_sessions
        ? `${st.completed_sessions} session(s) • ${st.num_interactions} interactions`
        : `${st.num_interactions} interactions`;

      tdStudent.innerHTML = `
        <div style="font-weight: 700; color: var(--accent-blue); display: flex; align-items: center; justify-content: space-between;">
          <span>${st.student_id}</span>
          <span style="font-size: 0.65rem; background: #eff6ff; padding: 1px 5px; border-radius: 4px; font-weight: normal;">View 🔍</span>
        </div>
        <div style="font-size: 0.68rem; color: var(--text-muted); font-weight: normal;">
          ${sessInfo}
        </div>
      `;
      tdStudent.title = `Click to view ${st.student_id}'s Skill Graph and Mastery Trajectory`;
      tdStudent.addEventListener('click', () => openStudentDetailModal(st.student_id, st, skills));
      tr.appendChild(tdStudent);

      // Skill cells (Task 10: data sufficiency indicators)
      skills.forEach(sk => {
        const td = document.createElement('td');
        const masteryVal = st.mastery ? st.mastery[sk.skill_id] : undefined;
        const counts = st.skill_interaction_counts || {};
        const intCount = counts[sk.skill_id] !== undefined ? counts[sk.skill_id] : (masteryVal !== undefined ? 1 : 0);
        const isLowData = masteryVal !== undefined && (intCount < 3 || (st.low_confidence_skills && st.low_confidence_skills.includes(sk.skill_id)));

        if (masteryVal !== undefined && masteryVal !== null) {
          const pct = (masteryVal * 100).toFixed(0);

          if (isLowData) {
            td.className = 'cell-low-data ';
            td.innerHTML = `
              <div>${pct}%</div>
              <div style="font-size: 0.62rem; color: #92400e; font-weight: normal;">(${intCount} ans)</div>
            `;
          } else {
            td.innerHTML = `
              <div>${pct}%</div>
              <div style="font-size: 0.62rem; color: var(--text-muted); font-weight: normal;">(${intCount} ans)</div>
            `;
          }

          td.title = `${st.student_id} on ${sk.name} (ID: ${sk.skill_id})\nMastery: ${(masteryVal * 100).toFixed(1)}%\nInteractions: ${intCount}${isLowData ? ' (⚠️ Low Data: <3 answers)' : ''}`;

          if (masteryVal >= 0.75) {
            td.className += 'cell-high';
          } else if (masteryVal >= 0.50) {
            td.className += 'cell-med';
          } else {
            td.className += 'cell-low';
          }
        } else {
          td.textContent = '—';
          td.className = 'empty-cell';
          td.title = `${st.student_id} has no recorded interactions for ${sk.name}`;
        }

        td.addEventListener('click', () => openStudentDetailModal(st.student_id, st, skills));
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
        if (st.mastery && st.mastery[sk.skill_id] !== undefined) {
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

  } catch (err) {
    console.error('Dashboard load error:', err);
    loadingIndicator.style.display = 'none';
    emptyState.style.display = 'block';
    emptyState.innerHTML = `<p style="color: var(--accent-rose);">Failed to load dashboard data: ${err.message}</p>`;
  }
}

let modalNetworkInstance = null;
let modalTrendChartInstance = null;

async function openStudentDetailModal(studentId, stData, skillsList) {
  const modal = document.getElementById('studentModal');
  const closeBtn = document.getElementById('modalCloseBtn');
  const title = document.getElementById('modalStudentTitle');
  const subtitle = document.getElementById('modalStudentSubtitle');
  const graphContainer = document.getElementById('modalSkillGraphVis');
  const canvas = document.getElementById('modalTrendCanvas');
  const singleNote = document.getElementById('modalSingleSessionNote');

  if (!modal) return;

  modal.style.display = 'flex';
  title.textContent = `Student Detail: ${studentId}`;
  subtitle.textContent = `${stData.completed_sessions || 1} completed session(s) • ${stData.num_interactions || 0} total interactions recorded`;

  closeBtn.onclick = () => { modal.style.display = 'none'; };
  modal.onclick = (e) => { if (e.target === modal) modal.style.display = 'none'; };

  // 1. Render PPMI Skill-Relation Graph for this student (Task 8 & 10)
  try {
    const graphRes = await fetch(`/api/graph/subgraph?student_id=${encodeURIComponent(studentId)}`);
    if (graphRes.ok) {
      const graphData = await graphRes.json();
      const { nodes, edges } = graphData;

      // Task 19: Display auto-generated mastery summary in modal
      const modalSummaryCard = document.getElementById('modalMasterySummary');
      const modalSummaryText = document.getElementById('modalMasterySummaryText');
      if (modalSummaryCard && modalSummaryText) {
        if (graphData.mastery_summary) {
          modalSummaryText.textContent = graphData.mastery_summary;
          modalSummaryCard.style.display = 'block';
        } else {
          modalSummaryCard.style.display = 'none';
        }
      }

      const nodeNameMap = {};
      (nodes || []).forEach(n => { nodeNameMap[n.raw_skill_id] = n.name; });

      const visNodes = (nodes || []).map(n => {
        const m = n.mastery !== undefined ? n.mastery : (stData.mastery && stData.mastery[String(n.raw_skill_id)]) || 0.5;
        const pct = Math.round(m * 100);
        const isWeak = m < 0.50;
        const cnt = n.interaction_count !== undefined ? n.interaction_count : 1;
        const isLowData = n.is_low_data || cnt < 3;

        let nodeBg = '#10b981';
        let borderCol = '#059669';
        if (isWeak) {
          nodeBg = '#f43f5e';
          borderCol = '#be123c';
        } else if (pct < 75) {
          nodeBg = '#f59e0b';
          borderCol = '#d97706';
        }

        return {
          id: n.raw_skill_id,
          label: `#${n.position} ${n.name.length > 14 ? n.name.slice(0, 12) + '…' : n.name}\n${pct}%${isLowData ? ' (⚠️)' : ''}`,
          title: createNodeTooltip(n.name, pct, cnt, isLowData),
          shape: 'box',
          margin: 8,
          color: { background: nodeBg, border: isLowData ? '#f59e0b' : borderCol },
          font: { color: '#ffffff', size: 10, face: 'sans-serif' },
          shapeProperties: { borderDashes: isLowData ? [3, 3] : false, borderRadius: 6 }
        };
      });

      const visEdges = (edges || []).map(e => {
        const w = e.weight || 0.5;
        const srcName = nodeNameMap[e.source] || `Skill #${e.source}`;
        const tgtName = nodeNameMap[e.target] || `Skill #${e.target}`;
        return {
          from: e.source,
          to: e.target,
          value: w,
          title: createEdgeTooltip(srcName, tgtName, w),
          width: Math.min(5, Math.max(1, (w / 2.5) * 2)),
          color: { color: '#cbd5e1', opacity: Math.min(0.9, Math.max(0.2, w / 9)) }
        };
      });

      if (window.vis && window.vis.Network) {
        if (modalNetworkInstance) modalNetworkInstance.destroy();
        modalNetworkInstance = new vis.Network(
          graphContainer,
          { nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) },
          {
            physics: {
              solver: 'forceAtlas2Based',
              forceAtlas2Based: { gravitationalConstant: -30, springLength: 80 },
              stabilization: { iterations: 100 }
            }
          }
        );
      }

      // Task 16: Render weak-skill relational callouts in modal
      const modalWeakCard = document.getElementById('modalWeakCallouts');
      const modalWeakList = document.getElementById('modalWeakCalloutsList');
      const callouts = graphData.weak_relational_callouts || [];

      if (modalWeakCard && modalWeakList) {
        if (callouts.length > 0) {
          modalWeakList.innerHTML = callouts.map(c => `<div style="margin-bottom: 2px;">• ${c}</div>`).join('');
          modalWeakCard.style.display = 'block';
        } else {
          modalWeakCard.style.display = 'none';
        }
      }
    }
  } catch (err) {
    console.warn('Modal graph error:', err);
  }

  // 2. Render Mastery Trend (Task 9)
  try {
    const trendRes = await fetch(`/api/student/${encodeURIComponent(studentId)}/trend`);
    if (trendRes.ok) {
      const trendData = await trendRes.json();
      const trend = trendData.trend || [];

      if (trend.length <= 1) {
        if (singleNote) singleNote.style.display = 'block';
      } else {
        if (singleNote) singleNote.style.display = 'none';
      }

      if (window.Chart && canvas) {
        const colors = [
          '#2563eb', '#0891b2', '#059669', '#16a34a',
          '#65a30d', '#d97706', '#ea580c', '#e11d48',
          '#9333ea', '#4f46e5'
        ];
        const labels = trend.map((t, i) => `S${t.session_index || i + 1}`);

        const datasets = (skillsList || []).map((sk, idx) => {
          const rawIdStr = String(sk.skill_id);
          const points = trend.map(t => {
            const val = (t.mastery && t.mastery[rawIdStr] !== undefined) ? t.mastery[rawIdStr] : 0.5;
            return Math.round(val * 100);
          });
          return {
            label: sk.name,
            data: points,
            borderColor: colors[idx % colors.length],
            backgroundColor: colors[idx % colors.length],
            borderWidth: 2,
            tension: 0.15,
            pointRadius: 4
          };
        });

        if (modalTrendChartInstance) modalTrendChartInstance.destroy();
        const ctx = canvas.getContext('2d');
        modalTrendChartInstance = new Chart(ctx, {
          type: 'line',
          data: { labels, datasets },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { min: 0, max: 100, ticks: { callback: v => `${v}%` } }
            },
            plugins: {
              legend: { display: false }
            }
          }
        });
      }
    }
  } catch (err) {
    console.warn('Modal trend error:', err);
  }
}

/**
 * Task 18 & Vis Network Tooltip Bug Fix:
 * Vis Network treats string `title` properties as plain text, escaping HTML and rendering `<br>` literally.
 * Returning an actual DOM HTMLElement (div) allows rich styling (bold headings, badges, custom spacing)
 * without any raw HTML tag leaks.
 */
function createNodeTooltip(name, masteryPct, count, isLowData) {
  const container = document.createElement('div');
  container.style.padding = '8px 12px';
  container.style.fontSize = '12px';
  container.style.lineHeight = '1.4';
  container.style.color = '#1e293b';

  const titleEl = document.createElement('div');
  titleEl.style.fontWeight = 'bold';
  titleEl.style.fontSize = '13px';
  titleEl.style.marginBottom = '4px';
  titleEl.style.color = '#0f172a';
  titleEl.textContent = name;
  container.appendChild(titleEl);

  const masteryRow = document.createElement('div');
  masteryRow.innerHTML = `<strong>Predicted Mastery:</strong> ${masteryPct}%`;
  container.appendChild(masteryRow);

  const countRow = document.createElement('div');
  countRow.innerHTML = `<strong>Interactions:</strong> ${count}`;
  container.appendChild(countRow);

  if (isLowData) {
    const noteRow = document.createElement('div');
    noteRow.style.color = '#b45309';
    noteRow.style.marginTop = '4px';
    noteRow.style.fontWeight = '600';
    noteRow.textContent = '⚠️ Low Data (<3 answers): estimate is preliminary';
    container.appendChild(noteRow);
  }

  return container;
}

function createEdgeTooltip(sourceName, targetName, weight) {
  const container = document.createElement('div');
  container.style.padding = '8px 12px';
  container.style.fontSize = '12px';
  container.style.lineHeight = '1.4';
  container.style.color = '#1e293b';

  const titleEl = document.createElement('div');
  titleEl.style.fontWeight = 'bold';
  titleEl.style.fontSize = '13px';
  titleEl.style.marginBottom = '4px';
  titleEl.style.color = '#0f172a';
  titleEl.textContent = 'Curriculum Relation';
  container.appendChild(titleEl);

  const pairRow = document.createElement('div');
  pairRow.textContent = `${sourceName} ↔ ${targetName}`;
  pairRow.style.fontWeight = '600';
  container.appendChild(pairRow);

  const weightRow = document.createElement('div');
  weightRow.innerHTML = `<strong>PPMI Co-occurrence Weight:</strong> ${Number(weight).toFixed(2)}`;
  container.appendChild(weightRow);

  return container;
}
