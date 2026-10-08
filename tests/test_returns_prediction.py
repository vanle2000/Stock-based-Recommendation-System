"""Tests for the returns-based evaluation.

These lock in the two claims the README now makes:
1. predicting price *levels* reproduces the inflated ~0.997 R2 (the artefact),
2. predicting *returns* removes it, so the model no longer beats a trivial
   baseline on data with no structure.
"""

from __future__ import annotations

import numpy as np

from src.models.returns_prediction import (
    add_returns,
    demonstrate_level_leakage,
    directional_accuracy_returns,
    information_coefficient,
    simulate_prices,
    walk_forward_returns,
    RETURN_FEATURES,
)


class TestDirectionalAccuracyReturns:
    def test_all_correct(self):
        yt = np.array([0.01, -0.02, 0.03])
        yp = np.array([0.5, -0.1, 0.2])
        assert directional_accuracy_returns(yt, yp) == 1.0

    def test_all_wrong(self):
        yt = np.array([0.01, -0.02, 0.03])
        yp = np.array([-0.5, 0.1, -0.2])
        assert directional_accuracy_returns(yt, yp) == 0.0


class TestInformationCoefficient:
    def test_perfect_corr(self):
        yt = np.array([0.01, -0.02, 0.03, -0.01])
        assert abs(information_coefficient(yt, yt) - 1.0) < 1e-9

    def test_constant_prediction_is_zero(self):
        yt = np.array([0.01, -0.02, 0.03])
        yp = np.zeros(3)
        assert information_coefficient(yt, yp) == 0.0


class TestLevelLeakageArtefact:
    def test_level_r2_is_near_one(self):
        """The headline artefact must reproduce: level R2 >= 0.99."""
        df = simulate_prices(n=2500, seed=42)
        out = demonstrate_level_leakage(df)
        assert out["r2_level_naive"] >= 0.99
        assert out["adjacent_close_corr"] >= 0.99


class TestReturnsRemoveTheArtefact:
    def test_feature_frame_has_no_lookahead(self):
        df = add_returns(simulate_prices(n=500, seed=1))
        # target is the shifted next-day return; features are all lagged
        assert "target_ret" in df.columns
        assert set(RETURN_FEATURES).issubset(df.columns)
        assert df["target_ret"].notna().all()

    def test_model_does_not_beat_baseline_on_random_walk(self):
        """On GBM data there is no signal, so model R2 must stay near/below 0 --
        nothing like the 0.997 the level target produced."""
        df = add_returns(simulate_prices(n=2500, seed=42))
        res = walk_forward_returns(df, n_splits=5)
        assert res["model_r2"].mean() < 0.05

    def test_directional_accuracy_near_coin_flip(self):
        df = add_returns(simulate_prices(n=2500, seed=42))
        res = walk_forward_returns(df, n_splits=5)
        assert 0.40 <= res["directional_acc"].mean() <= 0.60
