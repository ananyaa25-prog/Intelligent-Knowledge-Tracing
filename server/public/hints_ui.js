/**
 * hints_ui.js (v3)
 * Hint button and retry message for the student quiz. Inactive unless the backend
 * runs with HINTS_ENABLED=true and/or MAX_ATTEMPTS > 1 (the /next payload then
 * carries hints_enabled / max_attempts). Hooks into window.GKT like sections_ui.js.
 */
(function () {
  'use strict';

  var sessionId = null;
  var hintRow = document.getElementById('hintRow');
  var hintBtn = document.getElementById('hintBtn');
  var hintBox = document.getElementById('hintBox');
  var attemptInfo = document.getElementById('attemptInfo');
  var retryMsg = document.getElementById('retryMsg');
  if (!hintRow || !hintBtn || !hintBox) return;

  function show(el, on) { if (el) el.style.display = on ? '' : 'none'; }

  function onQuestion(q) {
    var hints = !!(q && q.hints_enabled);
    var maxAttempts = (q && q.max_attempts) || 1;
    show(hintBox, false);
    hintBox.textContent = '';
    show(retryMsg, false);
    hintBtn.disabled = false;
    show(hintBtn, hints);
    attemptInfo.textContent = maxAttempts > 1 ? 'Up to ' + maxAttempts + ' tries' : '';
    show(hintRow, hints || maxAttempts > 1);
  }

  function onRetry(a) {
    var left = a.attempts_left || 0;
    retryMsg.textContent = 'Not quite. Try again (' + left + (left === 1 ? ' try' : ' tries') + ' left).';
    show(retryMsg, true);
    attemptInfo.textContent = 'Try ' + (a.attempts_used + 1);
  }

  hintBtn.addEventListener('click', function () {
    if (!sessionId) return;
    hintBtn.disabled = true;
    fetch('/api/session/' + encodeURIComponent(sessionId) + '/hint', { method: 'POST' })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (res) {
        if (!res.ok) throw new Error((res.d && (res.d.detail || res.d.error)) || 'Hint unavailable');
        hintBox.textContent = 'Hint: ' + res.d.hint_text;
        show(hintBox, true);
      })
      .catch(function (e) {
        hintBox.textContent = 'Hint unavailable: ' + e.message;
        show(hintBox, true);
        hintBtn.disabled = false;
      });
  });

  window.GKT = window.GKT || {};
  var origReset = window.GKT.resetSession;
  window.GKT.resetSession = function (id) {
    sessionId = id;
    if (typeof origReset === 'function') return origReset.apply(this, arguments);
  };
  var origQuestion = window.GKT.onQuestion;
  window.GKT.onQuestion = function (q) {
    var r = typeof origQuestion === 'function' ? origQuestion.apply(this, arguments) : undefined;
    try { onQuestion(q); } catch (e) { /* optional UI */ }
    return r;
  };
  window.GKT.onRetry = function (a) { try { onRetry(a); } catch (e) { /* optional UI */ } };
})();
