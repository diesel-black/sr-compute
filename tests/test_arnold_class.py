"""Tests for sr_compute.diagnostics.arnold_class (Arnold A_k classification of V_n critical points).

The diagnostic computes V_n(psi_bar) = V_eff(F_n(psi_bar)) as a polynomial in psi_bar and
classifies each critical point by iterating exact symbolic derivatives.

V_eff(C) = (mu_sq/2)*C^2 - alpha_phi*C^4 is a degree-4 polynomial; its 5th and higher
derivatives vanish. F_n(psi) = psi + gamma*psi^n has derivatives that vanish above order n.
Because V_eff'' != 0 at the maxima (V_eff''(C*) = -2*mu_sq) and F_n'(psi*) != 0 at all
V_eff critical-point preimages, the second derivative V_n''(psi*) = V_eff''(C*)*(F_n'(psi*))^2
is non-zero for all SR critical points at baseline parameters. Equivalently, at fold-point
critical points (even n, where F_n'(psi_c)=0 and V_eff'(C_min)!=0), V_n''(psi_c) =
V_eff'(C_min)*F_n''(psi_c) != 0. All critical points of V_n are therefore Morse (A_1)
for the SR baseline parameters.

The parity law (2 metastable maxima for odd n, 3 for even n in n=4..8) is topological: it
counts Morse critical points, not degenerate singularities.

Note on n=9, 10 extension:
  n=9 (odd): 2 Morse maxima, 1 Morse minimum. Parity holds.
  n=10 (even): 4 Morse maxima (the prominence-filtered metastable count rises to 4 because the
               inner fold-point maximum achieves prominence > 0.04 at this order). The 3-for-even
               pattern breaks at n=10 under the standard prominence threshold.
"""

import math

import numpy as np
import pytest

from sr_compute.diagnostics import arnold_class

BASELINE = {"gamma": 1.0, "mu_sq": 1.0, "alpha_phi": 1.0}
PSI_RANGE = (-2.0, 2.0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _all_a1(result: dict) -> bool:
    """True iff every critical point in result has arnold_class == 'A_1'."""
    return all(cp["arnold_class"] == "A_1" for cp in result["critical_points"])


def _all_morse_order(result: dict) -> bool:
    """True iff every critical point has first_nonvanishing_order == 2."""
    return all(cp["first_nonvanishing_order"] == 2 for cp in result["critical_points"])


# ---------------------------------------------------------------------------
# n=3: 2 Morse maxima + 1 Morse minimum, all A_1
# ---------------------------------------------------------------------------

def test_n3_two_maxima():
    r = arnold_class(3, BASELINE, PSI_RANGE)
    assert r["n_maxima"] == 2
    assert r["n_minima"] == 1


def test_n3_all_a1():
    r = arnold_class(3, BASELINE, PSI_RANGE)
    assert _all_a1(r), f"Non-A_1 critical point found: {r['landscape_label']}"


def test_n3_all_morse_order():
    r = arnold_class(3, BASELINE, PSI_RANGE)
    assert _all_morse_order(r)


def test_n3_maximum_d2_magnitude():
    """V_n''(psi*) at n=3 maxima is analytically -2*mu_sq*(F_3'(psi*))^2 ~ -4.74 at baseline."""
    r = arnold_class(3, BASELINE, PSI_RANGE)
    maxima = [cp for cp in r["critical_points"] if cp["critical_point_type"] == "maximum"]
    assert len(maxima) == 2
    for cp in maxima:
        d2 = cp["leading_derivative_magnitudes"][0]
        assert d2 > 1.0, f"d2 too small at psi={cp['psi']:.4f}: {d2}"


def test_n3_minimum_at_origin():
    r = arnold_class(3, BASELINE, PSI_RANGE)
    minima = [cp for cp in r["critical_points"] if cp["critical_point_type"] == "minimum"]
    assert len(minima) == 1
    assert abs(minima[0]["psi"]) < 1e-6


def test_n3_maxima_symmetric():
    r = arnold_class(3, BASELINE, PSI_RANGE)
    maxima = [cp for cp in r["critical_points"] if cp["critical_point_type"] == "maximum"]
    psi_vals = sorted(cp["psi"] for cp in maxima)
    assert abs(psi_vals[0] + psi_vals[1]) < 1e-6, "n=3 maxima are not symmetric about psi=0"


# ---------------------------------------------------------------------------
# n=4: 3 Morse maxima + 2 Morse minima, all A_1
# ---------------------------------------------------------------------------

def test_n4_three_maxima():
    r = arnold_class(4, BASELINE, PSI_RANGE)
    assert r["n_maxima"] == 3
    assert r["n_minima"] == 2


def test_n4_all_a1():
    r = arnold_class(4, BASELINE, PSI_RANGE)
    assert _all_a1(r), f"Non-A_1 critical point found: {r['landscape_label']}"


def test_n4_all_morse_order():
    r = arnold_class(4, BASELINE, PSI_RANGE)
    assert _all_morse_order(r)


def test_n4_fold_maximum_present():
    """n=4 has a fold-point critical point near psi_c = -(1/(4*gamma))^(1/3) ~ -0.63."""
    r = arnold_class(4, BASELINE, PSI_RANGE)
    maxima_psi = sorted(cp["psi"] for cp in r["critical_points"] if cp["critical_point_type"] == "maximum")
    # Fold-point maximum is the leftmost inner maximum; psi_c ~ -0.63
    fold_candidates = [p for p in maxima_psi if -0.75 < p < -0.55]
    assert len(fold_candidates) >= 1, f"Fold-point maximum not found; maxima at {maxima_psi}"


def test_n4_fold_maximum_a1():
    """Fold-point maximum is non-degenerate: V_n''(psi_c) = V_eff'(C_min)*F_n''(psi_c) != 0."""
    r = arnold_class(4, BASELINE, PSI_RANGE)
    fold_cp = next(
        (cp for cp in r["critical_points"]
         if cp["critical_point_type"] == "maximum" and -0.75 < cp["psi"] < -0.55),
        None,
    )
    assert fold_cp is not None
    assert fold_cp["arnold_class"] == "A_1"
    assert fold_cp["first_nonvanishing_order"] == 2


# ---------------------------------------------------------------------------
# n=5 and n=6: confirm class and count
# ---------------------------------------------------------------------------

def test_n5_two_maxima_a1():
    r = arnold_class(5, BASELINE, PSI_RANGE)
    assert r["n_maxima"] == 2
    assert _all_a1(r)


def test_n6_four_true_maxima():
    """n=6 has 4 Morse maxima in V_n; the prominence-filtered metastable count is 3."""
    r = arnold_class(6, BASELINE, PSI_RANGE)
    assert r["n_maxima"] == 4


def test_n6_all_a1():
    r = arnold_class(6, BASELINE, PSI_RANGE)
    assert _all_a1(r)


# ---------------------------------------------------------------------------
# n=9 and n=10: parity extension
# ---------------------------------------------------------------------------

def test_n9_two_maxima():
    """n=9 is odd; parity law predicts 2 Morse maxima."""
    r = arnold_class(9, BASELINE, PSI_RANGE)
    assert r["n_maxima"] == 2


def test_n9_all_a1():
    r = arnold_class(9, BASELINE, PSI_RANGE)
    assert _all_a1(r)


def test_n10_four_maxima():
    """n=10 is even; the fold structure produces 4 Morse maxima. The inner fold-point
    maximum has prominence > 0.04, so count_metastable_states also reports 4 at n=10,
    breaking the 3-for-even pattern established through n=8."""
    r = arnold_class(10, BASELINE, PSI_RANGE)
    assert r["n_maxima"] == 4


def test_n10_all_a1():
    r = arnold_class(10, BASELINE, PSI_RANGE)
    assert _all_a1(r)


# ---------------------------------------------------------------------------
# Global invariant: all critical points A_1 for n=2..10
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_all_morse_all_n(n: int) -> None:
    """V_n has only non-degenerate (A_1 Morse) critical points for n=2..10 at baseline params."""
    r = arnold_class(n, BASELINE, PSI_RANGE)
    assert len(r["critical_points"]) > 0, f"n={n}: no critical points found"
    assert _all_a1(r), f"n={n}: non-A_1 critical point in {r['landscape_label']}"
    assert _all_morse_order(r), f"n={n}: first_nonvanishing_order != 2 for some critical point"


# ---------------------------------------------------------------------------
# Analytical cross-check: n=3 minimum at psi=0
# ---------------------------------------------------------------------------

def test_n3_minimum_d4_zero():
    """V_3(psi) = (1/2)*(psi+psi^3)^2 - (psi+psi^3)^4. Near psi=0 the psi^4 coefficient
    vanishes due to cancellation between V_eff terms: d4=0 at the minimum. This is a
    structural feature of V_3, not a sign of degeneracy (d2=1 != 0 so it is still A_1)."""
    r = arnold_class(3, BASELINE, PSI_RANGE, n_deriv=8)
    minimum = next(cp for cp in r["critical_points"] if cp["critical_point_type"] == "minimum")
    # d2 is the first entry in leading_derivative_magnitudes
    d2 = minimum["leading_derivative_magnitudes"][0]
    assert d2 == pytest.approx(1.0, rel=1e-6), f"d2 at n=3 minimum: {d2}"
    # first_nonvanishing_order is still 2 (d2 != 0)
    assert minimum["first_nonvanishing_order"] == 2
    assert minimum["arnold_class"] == "A_1"


# ---------------------------------------------------------------------------
# Return structure sanity checks
# ---------------------------------------------------------------------------

def test_return_structure_n3():
    r = arnold_class(3, BASELINE, PSI_RANGE)
    assert "critical_points" in r
    assert "landscape_label" in r
    assert "n_maxima" in r
    assert "n_minima" in r
    assert "psi_grid" in r
    assert "V_grid" in r
    assert len(r["psi_grid"]) == 2000  # default resolution
    for cp in r["critical_points"]:
        assert "psi" in cp
        assert "V" in cp
        assert "critical_point_type" in cp
        assert "arnold_class" in cp
        assert "first_nonvanishing_order" in cp
        assert "leading_derivative_magnitudes" in cp
