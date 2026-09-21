/**
 * server/tests/test_dashboard_smoke.js
 * ------------------------------------
 * Smoke test suite for Part B Teacher Dashboard.
 *
 * Verifies:
 * 1. Dashboard data contains >= 10 real test-split students with no crashes,
 *    no NaN cells, and valid skill columns.
 * 2. Spot-checks 3 student-skill cells against direct calls to Task 1 FastAPI
 *    service to confirm no silent transformation or mis-mapping.
 * 3. Asserts required honest demo disclaimer is present.
 */

const assert = require('assert');

const NODE_URL = 'http://127.0.0.1:3000';
const FASTAPI_URL = 'http://127.0.0.1:8000';

async function runDashboardSmokeTests() {
  console.log('=== RUNNING PART B (TEACHER DASHBOARD) SMOKE TESTS ===\n');

  // 1. Fetch dashboard matrix
  console.log('[Step 1] Fetching dashboard matrix from', `${NODE_URL}/api/dashboard/matrix`);
  const res = await fetch(`${NODE_URL}/api/dashboard/matrix`);
  assert.strictEqual(res.status, 200, `Dashboard endpoint failed with ${res.status}`);
  const data = await res.json();

  // Verify honest labeling
  assert(data.description, 'Missing description field');
  assert(
    data.description.includes('not a research contribution'),
    `Description must state 'not a research contribution': ${data.description}`
  );
  console.log('✓ PASS: Honest disclaimer verified in response payload.');

  const { students, skills } = data;

  // Requirement 1: At least 10 real test-split students
  assert(students.length >= 10, `Expected at least 10 students, got ${students.length}`);
  assert(skills.length > 0, 'Expected non-empty skills list');
  console.log(`✓ PASS: Received matrix with ${students.length} students across ${skills.length} skills.`);

  // Verify no NaNs or invalid values
  let testedCellsCount = 0;
  for (const st of students) {
    assert(st.student_id, 'Missing student_id');
    assert(st.num_interactions > 0, `Invalid interaction count for ${st.student_id}`);
    
    for (const [skillId, val] of Object.entries(st.mastery)) {
      assert(typeof val === 'number', `Val for ${st.student_id} on skill ${skillId} is not number: ${val}`);
      assert(!isNaN(val), `NaN detected for ${st.student_id} on skill ${skillId}`);
      assert(val >= 0.0 && val <= 1.0, `Out-of-bounds mastery for ${st.student_id} on skill ${skillId}: ${val}`);
      testedCellsCount++;
    }
  }
  console.log(`✓ PASS: All ${testedCellsCount} active student-skill cells are valid floats in [0, 1] with NO NaNs.\n`);

  // Requirement 2: Spot-check 2-3 cells against direct FastAPI call
  console.log('[Step 2] Spot-checking 3 cells against direct Task 1 FastAPI service calls...');

  const spotChecks = [
    { student_id: 'Student_01', skill_id: '277' },
    { student_id: 'Student_02', skill_id: '13' },
    { student_id: 'Student_03', skill_id: '15' }
  ];

  for (const check of spotChecks) {
    // A. Get dashboard value
    const studentData = students.find(s => s.student_id === check.student_id);
    assert(studentData, `Student ${check.student_id} not found in dashboard`);
    const dashboardVal = studentData.mastery[check.skill_id];
    assert(dashboardVal !== undefined, `Skill ${check.skill_id} not present for ${check.student_id}`);

    // B. Fetch student raw history
    const historyRes = await fetch(`${NODE_URL}/api/sample_students/${check.student_id}`);
    assert.strictEqual(historyRes.status, 200);
    const historyData = await historyRes.json();

    // C. Call FastAPI directly
    const fastApiRes = await fetch(`${FASTAPI_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ student_history: historyData.history })
    });
    assert.strictEqual(fastApiRes.status, 200);
    const fastApiData = await fastApiRes.json();
    const directVal = fastApiData.skill_mastery[check.skill_id];

    assert(directVal !== undefined, `Direct FastAPI call missing skill ${check.skill_id}`);

    const diff = Math.abs(dashboardVal - directVal);
    console.log(`  Spot-check [${check.student_id}, Skill ${check.skill_id}]:`);
    console.log(`    Dashboard value: ${dashboardVal.toFixed(6)}`);
    console.log(`    Direct FastAPI:  ${directVal.toFixed(6)}`);
    console.log(`    Difference:      ${diff.toExponential(2)}`);

    assert(diff <= 1e-6, `Spot check failed! Difference ${diff} exceeds 1e-6`);
    console.log('    ✓ Matched identically!\n');
  }

  console.log('==============================================');
  console.log('ALL PART B SMOKE TESTS PASSED SUCCESSFULLY!');
  console.log('==============================================\n');
}

runDashboardSmokeTests().catch(err => {
  console.error('\n❌ PART B SMOKE TEST FAILED:', err);
  process.exit(1);
});
