"""Unit tests for :class:`ValidityClassifier` (design §4 L4)."""
from __future__ import annotations

import numpy as np
import pytest

from bulbopt.optimization.learning.validity_classifier import (
    MIN_TRAINING_SAMPLES,
    ValidityClassifier,
)
from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)


# ---- helpers ---------------------------------------------------------------


def _linear_label_sample(
    n: int, seed: int = 0
) -> tuple[list[KrachtVector], list[int]]:
    """Draw ``n`` Kracht samples and label them by a simple linear rule.

    The rule: a candidate is "invalid" when the sum of its first two
    parameters exceeds a threshold. This gives logistic regression a
    learnable signal while keeping the test self-contained.
    """
    space = KrachtDesignSpace()
    samples = space.sample(n, seed=seed)
    # Separator set roughly at the midpoint of the feasible range.
    threshold = 0.5 * (
        space.bounds["length_ratio"][0]
        + space.bounds["length_ratio"][1]
        + space.bounds["breadth_ratio"][0]
        + space.bounds["breadth_ratio"][1]
    )
    labels = [
        int((v.values["length_ratio"] + v.values["breadth_ratio"]) > threshold)
        for v in samples
    ]
    return samples, labels


# ---- tests -----------------------------------------------------------------


def test_classifier_returns_none_below_min_samples():
    samples, labels = _linear_label_sample(MIN_TRAINING_SAMPLES - 1, seed=1)
    clf = ValidityClassifier()
    clf.fit(samples, labels)
    for sample in samples:
        assert clf.predict_invalid_probability(sample) is None


def test_classifier_returns_none_before_fit():
    clf = ValidityClassifier()
    space = KrachtDesignSpace()
    vector = space.sample(1, seed=0)[0]
    assert clf.predict_invalid_probability(vector) is None


def test_classifier_learns_correct_direction():
    """Predictions should be higher for invalid-labelled samples."""
    samples, labels = _linear_label_sample(80, seed=2)
    clf = ValidityClassifier()
    clf.fit(samples, labels)

    invalid_scores: list[float] = []
    valid_scores: list[float] = []
    for sample, label in zip(samples, labels):
        prob = clf.predict_invalid_probability(sample)
        assert prob is not None
        assert 0.0 <= prob <= 1.0
        if label == 1:
            invalid_scores.append(prob)
        else:
            valid_scores.append(prob)

    assert invalid_scores, "test fixture produced no invalid samples"
    assert valid_scores, "test fixture produced no valid samples"
    assert np.mean(invalid_scores) > np.mean(valid_scores)


def test_classifier_accepts_raw_arrays():
    """``fit`` should accept plain float lists alongside KrachtVector objects."""
    samples, labels = _linear_label_sample(40, seed=3)
    clf = ValidityClassifier()
    arrays = [[v.values[name] for name in KRACHT_PARAMETER_NAMES] for v in samples]
    clf.fit(arrays, labels)

    probe = samples[0]
    prob = clf.predict_invalid_probability(probe)
    assert prob is not None
    assert 0.0 <= prob <= 1.0


def test_classifier_single_class_history_returns_zero_probability():
    """All-valid history should not reject candidates — predictor pins at 0."""
    samples, _ = _linear_label_sample(MIN_TRAINING_SAMPLES + 5, seed=4)
    labels = [0] * len(samples)
    clf = ValidityClassifier()
    clf.fit(samples, labels)

    prob = clf.predict_invalid_probability(samples[0])
    assert prob == pytest.approx(0.0)


def test_classifier_does_not_freeze_when_all_history_is_invalid():
    """All-invalid history must NOT lock the predictor at p=1.0.

    Bug #4 (audit 2026-04-26): when the first 10 candidates all happen to
    be invalid (e.g. a bad baseline mesh on the first night-run), the
    classifier used to memorise ``float(1.0)`` and reject every future
    candidate forever, freezing NSGA-II in a flat fitness landscape. The
    safe behaviour is to either fall back to "no prediction" (None) or
    return a probability low enough that the prefilter does not reject
    everything.
    """
    samples, _ = _linear_label_sample(MIN_TRAINING_SAMPLES + 5, seed=11)
    labels = [1] * len(samples)
    clf = ValidityClassifier()
    clf.fit(samples, labels)

    prob = clf.predict_invalid_probability(samples[0])
    assert prob is None or prob <= 0.5, (
        f"all-invalid fit must not freeze at p=1.0 (got {prob!r})"
    )


def test_classifier_does_not_freeze_when_all_history_is_valid():
    """All-valid history → predictor returns 0.0 or None (no rejection)."""
    samples, _ = _linear_label_sample(MIN_TRAINING_SAMPLES + 5, seed=12)
    labels = [0] * len(samples)
    clf = ValidityClassifier()
    clf.fit(samples, labels)

    prob = clf.predict_invalid_probability(samples[0])
    assert prob is None or prob == pytest.approx(0.0)


def test_classifier_predicts_normally_when_both_classes_seen():
    """Two-class fit produces probabilities that span the (0, 1) range."""
    space = KrachtDesignSpace()
    samples = space.sample(20, seed=13)
    labels = [1] * 10 + [0] * 10
    clf = ValidityClassifier()
    clf.fit(samples, labels)

    probs = []
    for sample in samples:
        prob = clf.predict_invalid_probability(sample)
        assert prob is not None
        assert 0.0 <= prob <= 1.0
        probs.append(prob)

    # Predictions should span (not all clamped to 0 or 1).
    assert min(probs) < 0.9
    assert max(probs) > 0.1


def test_classifier_rejects_mismatched_lengths():
    samples, _ = _linear_label_sample(MIN_TRAINING_SAMPLES + 2, seed=5)
    clf = ValidityClassifier()
    with pytest.raises(ValueError):
        clf.fit(samples, labels=[0, 1, 0])


def test_classifier_predicts_from_raw_array():
    samples, labels = _linear_label_sample(40, seed=6)
    clf = ValidityClassifier()
    clf.fit(samples, labels)

    probe_array = [samples[0].values[name] for name in KRACHT_PARAMETER_NAMES]
    prob = clf.predict_invalid_probability(probe_array)
    assert prob is not None
    assert 0.0 <= prob <= 1.0


def test_classifier_rejects_wrong_dimension_vector():
    samples, labels = _linear_label_sample(40, seed=7)
    clf = ValidityClassifier()
    clf.fit(samples, labels)
    with pytest.raises(ValueError):
        clf.predict_invalid_probability([0.1, 0.2, 0.3])
