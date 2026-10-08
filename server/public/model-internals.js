// model-internals.js
// Logic for inspecting GatedKT internal gate dynamics and visualizations

document.addEventListener('DOMContentLoaded', () => {
  const sessionIdInput = document.getElementById('sessionIdInput');
  const inspectBtn = document.getElementById('inspectBtn');
  const loadRecentBtn = document.getElementById('loadRecentBtn');
  const sessionStatusMsg = document.getElementById('sessionStatusMsg');

  const gateVisualizerCard = document.getElementById('gateVisualizerCard');
  const sessionMetaBadge = document.getElementById('sessionMetaBadge');
  const gateInsufficientNote = document.getElementById('gateInsufficientNote');
  const gateActiveContent = document.getElementById('gateActiveContent');
  const gateStepsContainer = document.getElementById('gateStepsContainer');
  const gatePlotImg = document.getElementById('gatePlotImg');
  const gatePlotLoading = document.getElementById('gatePlotLoading');

  const errorStateCard = document.getElementById('errorStateCard');
  const errorTitle = document.getElementById('errorTitle');
  const errorMessage = document.getElementById('errorMessage');
  const retryBtn = document.getElementById('retryBtn');

  // Check URL query param ?session_id=...
  const urlParams = new URLSearchParams(window.location.search);
  const initialSessionId = urlParams.get('session_id');
  if (initialSessionId) {
    sessionIdInput.value = initialSessionId;
    inspectSession(initialSessionId);
  }

  inspectBtn.addEventListener('click', () => {
    const sid = sessionIdInput.value.trim();
    if (!sid) {
      showStatusMsg('Please enter a session ID.', true);
      return;
    }
    inspectSession(sid);
  });

  loadRecentBtn.addEventListener('click', async () => {
    showStatusMsg('Fetching latest completed session...', false);
    try {
      const res = await fetch('/api/dashboard/matrix?source=all');
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      const students = data.students || [];
      if (students.length === 0) {
        showStatusMsg('No student sessions recorded yet in database.', true);
        return;
      }

      // Find first student with completed sessions
      const st = students[0];
      const stId = st.student_id;

      // Get student's trend or history to find session_id
      const trendRes = await fetch(`/api/student/${encodeURIComponent(stId)}/trend`);
      if (trendRes.ok) {
        const trendData = await trendRes.json();
        const pts = trendData.trend || [];
        if (pts.length > 0 && pts[pts.length - 1].session_id) {
          const sid = pts[pts.length - 1].session_id;
          sessionIdInput.value = sid;
          showStatusMsg(`Loaded session for student '${stId}'.`, false);
          inspectSession(sid);
          return;
        }
      }

      showStatusMsg(`Found student '${stId}', but could not resolve session ID.`, true);
    } catch (err) {
      showStatusMsg(`Failed to fetch recent sessions: ${err.message}`, true);
    }
  });

  retryBtn.addEventListener('click', () => {
    const sid = sessionIdInput.value.trim();
    if (sid) inspectSession(sid);
  });

  function showStatusMsg(msg, isError) {
    sessionStatusMsg.textContent = msg;
    sessionStatusMsg.style.color = isError ? '#e11d48' : '#2563eb';
    sessionStatusMsg.style.display = 'block';
  }

  function showError(title, msg) {
    gateVisualizerCard.style.display = 'none';
    errorTitle.textContent = title;
    errorMessage.textContent = msg;
    errorStateCard.style.display = 'block';
  }

  async function inspectSession(sessionId) {
    errorStateCard.style.display = 'none';
    gateVisualizerCard.style.display = 'block';
    sessionMetaBadge.textContent = `Session: ${sessionId.slice(0, 8)}…`;
    gatePlotLoading.style.display = 'block';
    gatePlotImg.style.display = 'none';
    gateStepsContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">Loading step activations...</div>';

    try {
      const res = await fetch(`/api/session/${encodeURIComponent(sessionId)}/gate_detail`);
      if (!res.ok) {
        if (res.status === 404) {
          throw new Error(`Session '${sessionId}' was not found in the database.`);
        }
        throw new Error(`Server returned error status ${res.status}`);
      }

      const data = await res.json();
      if (!data.available) {
        gateActiveContent.style.display = 'none';
        gateInsufficientNote.style.display = 'block';
        gateInsufficientNote.textContent = data.message || 'Not enough interactions recorded yet for gate interpretability (at least 3 required).';
        return;
      }

      gateInsufficientNote.style.display = 'none';
      gateActiveContent.style.display = 'block';

      // Render step means
      const means = data.step_gate_means || [];
      gateStepsContainer.innerHTML = '';

      if (means.length === 0) {
        gateStepsContainer.innerHTML = '<div style="font-size: 0.82rem; color: var(--text-muted);">No step transitions recorded.</div>';
      } else {
        means.forEach((m, idx) => {
          const pill = document.createElement('div');
          pill.className = 'step-pill';

          const stepLabel = document.createElement('span');
          stepLabel.style.color = '#64748b';
          stepLabel.style.fontSize = '0.72rem';
          stepLabel.textContent = `Step ${idx + 1} → ${idx + 2}`;

          const valSpan = document.createElement('span');
          valSpan.className = 'step-pill-val';
          valSpan.textContent = m.toFixed(3);

          if (m >= 0.5) {
            valSpan.style.color = '#2563eb'; // Short-term heavy
          } else {
            valSpan.style.color = '#059669'; // Long-term/graph heavy
          }

          const biasLabel = document.createElement('span');
          biasLabel.style.fontSize = '0.68rem';
          biasLabel.style.color = '#64748b';
          biasLabel.textContent = m >= 0.5 ? 'Short-Term Bias' : 'Long-Term Bias';

          pill.appendChild(stepLabel);
          pill.appendChild(valSpan);
          pill.appendChild(biasLabel);
          gateStepsContainer.appendChild(pill);
        });
      }

      // Load plot image
      gatePlotImg.onload = () => {
        gatePlotLoading.style.display = 'none';
        gatePlotImg.style.display = 'inline-block';
      };
      gatePlotImg.onerror = () => {
        gatePlotLoading.textContent = 'Could not generate gate plot image for this session.';
      };
      gatePlotImg.src = `/api/session/${encodeURIComponent(sessionId)}/gate_plot.png?t=${Date.now()}`;

    } catch (err) {
      console.warn('Session inspection notice:', err);
      showError('Session Inspection Failed', err.message || 'Unable to connect to backend.');
    }
  }
});
