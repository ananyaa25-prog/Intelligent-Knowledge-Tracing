"""
tests/test_adaptive_session.py
------------------------------
Self-tests for the adaptive session engine (Phase B). Uses a deterministic fake
predictor so the selection logic can be asserted without the model.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serving.adaptive_session import (
    AdaptiveSession, AdaptiveSessionManager,
    MAIN_QUESTIONS, MAX_PER_TOPIC, TOPIC_IDS, LEVEL_TO_BUCKET,
)


def make_predictor(score_fn):
    def predictor(history, candidates):
        return [score_fn(topic, bucket) for topic, bucket in candidates]
    return predictor


def answer(sess, text="1"):
    sess.record_answer(text, 5000.0, 0)


def test_warmup_has_no_prediction():
    sess = AdaptiveSession("s1", "stu")
    q1 = sess.next_question(predictor=None)
    assert q1["predicted_p"] is None
    assert q1["difficulty_bucket"] == 2  # start medium
    answer(sess)
    q2 = sess.next_question(predictor=None)
    assert q2["predicted_p"] is None
    assert q2["difficulty_bucket"] == 2
    answer(sess)


def test_adaptive_picks_lowest_p():
    sess = AdaptiveSession("s1", "stu")
    sess.next_question(predictor=None); answer(sess)
    sess.next_question(predictor=None); answer(sess)

    pred = make_predictor(lambda topic, bucket: (topic % 10) / 10.0 + bucket * 0.01)
    q = sess.next_question(predictor=pred)
    assert q["predicted_p"] is not None
    eligible = [t for t in TOPIC_IDS if sess.topic_count[t] < MAX_PER_TOPIC]
    expected_topic = min(eligible, key=lambda t: pred([], [(t, LEVEL_TO_BUCKET[sess.levels[t]])])[0])
    assert q["topic"] == expected_topic


def test_staircase_moves_with_answer():
    sess = AdaptiveSession("s1", "stu")
    topic = TOPIC_IDS[0]
    # correct -> up
    sess.pending = sess._make_question(topic, "medium", False, None)
    sess.record_answer(sess.pending["correct_answer"], 5000.0)
    assert sess.levels[topic] == "hard"
    # wrong -> down
    sess.pending = sess._make_question(topic, "hard", False, None)
    sess.record_answer("definitely_wrong", 5000.0)
    assert sess.levels[topic] == "medium"
    # wrong again -> down to easy
    sess.pending = sess._make_question(topic, "medium", False, None)
    sess.record_answer("wrong", 5000.0)
    assert sess.levels[topic] == "easy"
    # wrong at easy stays clamped at easy
    sess.pending = sess._make_question(topic, "easy", False, None)
    sess.record_answer("wrong", 5000.0)
    assert sess.levels[topic] == "easy"
    # correct at hard stays clamped at hard
    sess.levels[topic] = "hard"
    sess.pending = sess._make_question(topic, "hard", False, None)
    sess.record_answer(sess.pending["correct_answer"], 5000.0)
    assert sess.levels[topic] == "hard"


def test_topic_caps_and_consecutive():
    sess = AdaptiveSession("s1", "stu")
    pred = make_predictor(lambda topic, bucket: 0.5)
    while len(sess.history) < MAIN_QUESTIONS:
        sess.next_question(predictor=pred)
        answer(sess)
        for t in TOPIC_IDS:
            assert sess.topic_count[t] <= MAX_PER_TOPIC, f"topic {t} exceeded cap"
    assert len(sess.history) == MAIN_QUESTIONS
    for i in range(2, len(sess.history)):
        three = [sess.history[j]["topic"] for j in range(i - 2, i + 1)]
        assert not (three[0] == three[1] == three[2]), "3 in a row on same topic"


def test_probe_questions_on_unasked_topics():
    sess = AdaptiveSession("s1", "stu")
    pred = make_predictor(lambda topic, bucket: 0.5)
    while len(sess.history) < MAIN_QUESTIONS:
        sess.next_question(predictor=pred)
        answer(sess)
    probe = sess.next_question(predictor=pred)
    assert probe is None or probe["is_probe"] is True
    if probe is not None:
        assert probe["topic"] not in sess.asked_topics


def test_manager_logs_and_stores():
    logs = []
    stored = []

    def store(session_id, student_id, inter):
        stored.append(inter)

    def log(rec):
        logs.append(rec)

    mgr = AdaptiveSessionManager(store=store, log=log)
    sid, st = mgr.start(student_id="stu")
    q = mgr.next_question(sid, predictor=None)
    assert q is not None
    inter = mgr.record_answer(sid, "42", 1000.0)
    assert inter["correct"] in (0, 1)
    assert any(e.get("event") == "answer" for e in logs)
    assert len(stored) == 1
    assert stored[0]["difficulty_bucket"] in (1, 2, 3)
    assert stored[0]["level"] in ("easy", "medium", "hard")


def test_full_session_runs():
    stored = []
    mgr = AdaptiveSessionManager(store=lambda *a: stored.append(a), log=lambda r: None)
    sid, st = mgr.start(student_id="stu")
    pred = make_predictor(lambda topic, bucket: 0.3 + 0.1 * (topic % 5))
    answered = 0
    while answered < 60:  # safety bound
        q = mgr.next_question(sid, predictor=pred)
        if q is None:
            break
        mgr.record_answer(sid, "1", 5000.0)
        answered += 1
    hist = mgr.history(sid)
    main = [h for h in hist if not h["is_probe"]]
    probes = [h for h in hist if h["is_probe"]]
    assert len(main) == MAIN_QUESTIONS, f"expected {MAIN_QUESTIONS} main questions, got {len(main)}"
    assert 0 <= len(probes) <= 2
    for h in main:
        assert h["p_before"] is not None or h["position"] <= 2  # only warm-up lacks a prediction


if __name__ == "__main__":
    test_warmup_has_no_prediction()
    test_adaptive_picks_lowest_p()
    test_staircase_moves_with_answer()
    test_topic_caps_and_consecutive()
    test_probe_questions_on_unasked_topics()
    test_manager_logs_and_stores()
    test_full_session_runs()
    print("All adaptive session tests passed.")
