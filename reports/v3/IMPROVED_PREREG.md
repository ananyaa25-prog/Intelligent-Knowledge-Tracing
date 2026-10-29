# Pre-registration: improved model (GatedKT+) benchmark

Written 2026-10-29 before any test-set result of GatedKT+ existed (only the validation-only tuning in
reports/v3/gkt_plus_tuning.md may run before this file is committed). Nothing below is changed after the results,
except bug fixes, which will be listed at the end of this file with their reason.

## What is tested

- **gkt_plus** (src/baselines/gkt_plus.py): GatedKT encoders + gate, Rasch-style per-question difficulty
  (embedding and logit offset, train questions only), no skill graph, dropout + weight decay chosen on the
  validation split (reports/v3/gkt_plus_tuning.json).
- **gkt_plus_elo**: gkt_plus blended with the Elo baseline by a 2-feature logistic regression fitted on the
  validation split only. **Primary model.**
- **gkt_plus_pt** (Statics2011 only): gkt_plus initialised from the ASSISTments-trained gkt_plus (seed 0) and
  fine-tuned.
- Comparators: full_gated (our original model), dkt_classic, dkt_gru, elo, logreg_counts.

## Protocol

- Same budget for every deep model: up to 15 epochs, early stopping on validation AUC with patience 3, Adam
  1e-3, batch 32, best-validation weights, 1 CPU thread per run.
- Datasets and seeds: ASSISTments official split, 20 seeds (0-19); Eedi primary and Eedi secondary (seeded
  80/10/10 of data/processed_v3/eedi.pkl), 5 seeds each; Statics2011 (seeded 80/10/10), 10 seeds.
- Statistics exactly as in Stage 5: per-seed test AUC; paired t-test over seeds with Holm correction over the
  comparisons of each dataset report; Wilcoxon; 1000-resample student bootstrap.
- Pre-registered comparisons: gkt_plus_elo vs dkt_classic, elo, full_gated, dkt_gru, logreg_counts, gkt_plus;
  gkt_plus vs dkt_classic, elo, full_gated; on Statics also gkt_plus_pt vs gkt_plus and vs elo.
- Verdict rule (unchanged from Stage 5): "better" only if the Holm-corrected p < 0.05 AND the seed CI and the
  bootstrap CI of the difference both exclude 0; "worse" symmetrically; else "comparable".

## Success criterion

The improved model is called "better than existing models" on a dataset only if gkt_plus_elo is "better" than
BOTH dkt_classic and elo there. If that holds on some datasets and not others, the claim is stated per dataset.
A "comparable" or "worse" verdict is reported as such. The app's served model is not changed by this benchmark.

## Not done

Time-since-last-attempt features (item 5 of the improvement plan) are not included: the ASSISTments pipeline does
not keep usable timestamps and only Eedi has answer dates.

## Bug fixes after pre-registration

None. The benchmark ran as pre-registered; results in reports/v3/IMPROVED_SUMMARY.md.
