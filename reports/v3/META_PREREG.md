# Pre-registration: hints and attempts inside the GRU and the gate (GatedKT+ meta)

Written 2026-10-30 before any test-set result of the models below existed. Nothing below is changed after the
results, except bug fixes, which will be listed at the end of this file with their reason.

## Why

In GatedKT and GatedKT+ the attempt count and hint flag reach the model only through a small metadata vector that
feeds the gate; the no-metadata ablation of the original model gained just +0.0010 AUC (reports/v3/benchmark.md).
In ASSISTments 2009 a hint request makes the first attempt count as wrong (P(correct | hint) = 0 in the train split)
and answers with 2+ attempts are almost always wrong (P(correct) 0.035 at 2 attempts without a hint), so hints and
attempts describe HOW a student got a question wrong. 14.7% of train interactions used a hint and 18.1% had more
than one attempt.

## What is tested (src/baselines/gkt_plus.py, meta_mode)

- **gkt_plus_meta** ("full", primary model): (a) GRU input uses a 7-way outcome embedding (correct; wrong with
  1 / 2 / 3+ attempts; each with or without a hint) instead of a binary correct embedding; (b) the metadata vector
  (response time, attempts, hint) is projected into the input of both GRUs; (c) the gate additionally sees the share
  of hints and of multi-attempt answers over the last 10 steps.
- **gkt_plus_meta_gru** ("gru"): (a) + (b), gate without metadata.
- **gkt_plus** ("gate", unchanged from IMPROVED_PREREG.md; its cached predictions for seeds 0-9 are reused).
- **gkt_plus_nometa** ("none"): no response time, attempts or hints anywhere.
- Comparators (cached from the IMPROVED run, same budget): dkt_classic, dkt_gru.

Only the history steps' hints and attempts are used; the predicted step's own hints / attempts never are (no leakage).

## Protocol

ASSISTments 2009 official split only (Eedi and Statics2011 have no hint or attempt data). Seeds 0-9. Same budget as
IMPROVED_PREREG.md: up to 15 epochs, early stopping on validation AUC with patience 3, Adam 1e-3, batch 32, dropout
0.3 and weight decay 1e-5 (chosen earlier on validation for gkt_plus; not re-tuned). Statistics, Holm correction over
the comparisons below, 1000-resample student bootstrap and the verdict rule are exactly as in IMPROVED_PREREG.md.

Comparisons: gkt_plus_meta vs gkt_plus, gkt_plus_nometa, gkt_plus_meta_gru, dkt_classic, dkt_gru;
gkt_plus_meta_gru vs gkt_plus_nometa; gkt_plus vs gkt_plus_nometa.

## Success criterion

"Putting hints and attempts into the GRU and the gate improves the model" is claimed only if gkt_plus_meta is
"better" than gkt_plus (gate-only metadata). Otherwise the result is reported as comparable or worse. Pathway effects
(GRU vs gate) are read from the ablation rows and are described as such, not as separate claims unless "better".

A descriptive check (no test, not a claim of correctness) is also reported: for the seed-0 gkt_plus_meta model, how
the predicted P(next correct) and the mean gate value change when the last answer of a test history is recorded as
wrong without help, wrong after 2 / 3 attempts, or wrong with a hint.

## Demo rule (added 2026-10-30, still before any test result of these models)

The app gets the hint button and retries behind HINTS_ENABLED / MAX_ATTEMPTS (defaults off). demo.env sets
SERVING_MODEL=gkt_plus_meta only if BOTH hold: (1) gkt_plus_meta is not "worse" than gkt_plus in
reports/v3/meta_assist.md, and (2) the saved seed-0 gkt_plus_meta checkpoint's ASSISTments test AUC is at least the
test AUC of the currently served checkpoint (data/processed/best_full_gated.pt), both computed by the same script.
Otherwise demo.env keeps SERVING_MODEL=gated_kt (that model still receives hints and attempts, but only through its
gate).

## Bug fixes after pre-registration

None. Results: reports/v3/meta_assist.md and reports/v3/meta_serving_check.md.
