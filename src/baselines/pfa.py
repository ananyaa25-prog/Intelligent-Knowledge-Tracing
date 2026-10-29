"""
src/baselines/pfa.py
--------------------
Performance Factors Analysis (Pavlik, Cen & Koedinger, 2009), fitted as a
logistic regression:

    logit P = beta_k + gamma_k * successes_k + rho_k * failures_k + delta_d

with per-skill intercept and per-skill weights on the RAW prior success / failure
counts on the target skill k, plus a difficulty-bucket term delta_d (added so all
baselines see the same difficulty buckets as the main model). L2 regularised (C=1).
"""

from baselines.logistic import LogisticCountsBaseline


class PFABaseline(LogisticCountsBaseline):
    name = "pfa"
    per_skill = True
