"""Learning helpers for night-run optimization.

Modules:

* ``validity_classifier`` — logistic regression on accumulated HF history to
  predict whether a candidate's mesh will be invalid (self-intersecting,
  non-watertight, zero/negative volume). Used as a prefilter by the mid-gate
  so obviously broken candidates never reach the deformer (see design spec
  2026-04-23 §4 L4).
* ``history_store`` — JSONL append-only store of completed HF evaluations
  (inputs → drag, mesh metrics, validity). Feeds both the GP surrogate (L1)
  and the validity classifier (L4).
* ``gp_surrogate`` — scikit-learn Gaussian-process regressor trained on the
  history store; warm-starts NSGA-II and orders the queue so the cheapest
  candidates are evaluated first (L1).

Design reference: 2026-04-23-bulbopt-mesh-quality-design.md §4 (L1 + L4).
"""
