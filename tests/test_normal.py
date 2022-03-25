# probcalc - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""A simple test module to test the :class:`probcalc.distributions.NormalDistribution`.

All test values calculated with a Casio fx-991EX Classwiz.
"""

import pytest
from pytest import approx

from probcalc import P, N, NonsenseError


def test_pmf() -> None:
    """Test the normal distribution PMF."""
    X = N(2, 0.5)
    Y = N(-3.9, 1.6)
    Z = N(0, 1)
    W = N(1000, 100)

    assert X.pmf(0) == approx(2.676604515e-4)
    assert X.pmf(1) == approx(0.107981933)
    assert X.pmf(-1) == approx(1.21517657e-8)
    assert X.pmf(2) == approx(0.7978845608)
    assert X.pmf(3) == approx(0.107981933)
    assert X.pmf(4) == approx(2.676604515e-4)
    assert X.pmf(2.3) == approx(0.6664492058)
    assert X.pmf(-0.5) == approx(2.973439029e-6)
    assert X.pmf(100) == 0

    assert Y.pmf(0) == approx(0.01278292111)
    assert Y.pmf(1) == approx(2.291851466e-3)
    assert Y.pmf(-1) == approx(0.04824224027)
    assert Y.pmf(-2) == approx(0.1231908762)
    assert Y.pmf(-3) == approx(0.2128547464)
    assert Y.pmf(-4) == approx(0.2488524104)
    assert Y.pmf(-5) == approx(0.1968584596)
    assert Y.pmf(-6) == approx(0.1053707403)
    assert Y.pmf(-3.9) == approx(0.2493389253)
    assert Y.pmf(-2.1) == approx(0.1324229036)
    assert Y.pmf(-5.3) == approx(0.170034374)
    assert Y.pmf(100) == 0

    assert Z.pmf(0) == approx(0.3989422804)
    assert Z.pmf(0.5) == approx(0.3520653268)
    assert Z.pmf(1) == approx(0.2419707245)
    assert Z.pmf(1.5) == approx(0.1295175957)
    assert Z.pmf(2) == approx(0.05399096651)
    assert Z.pmf(2.5) == approx(0.01752830049)
    assert Z.pmf(3) == approx(4.431848412e-3)
    assert Z.pmf(3.5) == approx(8.72682695e-4)
    assert Z.pmf(4) == approx(1.338302258e-4)
    assert Z.pmf(4.5) == approx(1.598374111e-5)
    assert Z.pmf(5) == approx(1.486719515e-6)
    assert Z.pmf(-0.5) == approx(0.3520653268)
    assert Z.pmf(-1) == approx(0.2419707245)
    assert Z.pmf(-1.5) == approx(0.1295175957)
    assert Z.pmf(-2) == approx(0.05399096651)
    assert Z.pmf(-2.5) == approx(0.01752830049)
    assert Z.pmf(-3) == approx(4.431848412e-3)
    assert Z.pmf(-3.5) == approx(8.72682695e-4)
    assert Z.pmf(-4) == approx(1.338302258e-4)
    assert Z.pmf(-4.5) == approx(1.598374111e-5)
    assert Z.pmf(-5) == approx(1.486719515e-6)

    assert W.pmf(100) == approx(1.027977357e-20)
    assert W.pmf(200) == approx(5.052271084e-17)
    assert W.pmf(300) == approx(9.134720408e-14)
    assert W.pmf(400) == approx(6.07588285e-11)
    assert W.pmf(500) == approx(1.486719515e-8)
    assert W.pmf(600) == approx(1.338302258e-6)
    assert W.pmf(700) == approx(4.431848412e-5)
    assert W.pmf(800) == approx(5.399096651e-4)
    assert W.pmf(900) == approx(2.419707245e-3)
    assert W.pmf(1000) == approx(3.989422804e-3)
    assert W.pmf(1100) == approx(2.419707245e-3)
    assert W.pmf(1200) == approx(5.399096651e-4)
    assert W.pmf(1300) == approx(4.431848412e-5)
    assert W.pmf(1400) == approx(1.338302258e-6)
    assert W.pmf(1500) == approx(1.486719515e-8)
    assert W.pmf(1600) == approx(6.07588285e-11)
    assert W.pmf(1700) == approx(9.134720408e-14)
    assert W.pmf(1800) == approx(5.052271084e-17)
    assert W.pmf(1900) == approx(1.027977357e-20)

    with pytest.raises(NonsenseError):
        N(1, 0)

    with pytest.raises(NonsenseError):
        N(1, -1)


def test_cdf() -> None:
    """Test the normal distribution CDF."""
    X = N(2, 0.5)
    Y = N(-3.9, 1.6)
    Z = N(0, 1)
    W = N(1000, 100)

    assert X.cdf(2) == 0.5
    assert Y.cdf(-3.9) == 0.5
    assert Z.cdf(0) == 0.5
    assert W.cdf(1000) == 0.5

    # We're using specific absolute tolerances for these tests because
    # normal cdf uses utility.erf, which uses a numerical approximation
    assert X.cdf(-3) == approx(7.619853024e-24, abs=1.5e-7)
    assert X.cdf(-2) == approx(6.220960574e-16, abs=1.5e-7)
    assert X.cdf(-1) == approx(9.865876418e-10, abs=1.5e-7)
    assert X.cdf(0) == approx(3.167126301e-5, abs=1.5e-7)
    assert X.cdf(1) == approx(0.022750132, abs=1.5e-7)
    assert X.cdf(1.5) == approx(0.1586552539, abs=1.5e-7)
    assert X.cdf(2) == 0.5
    assert X.cdf(2.5) == approx(0.8413447461, abs=1.5e-7)
    assert X.cdf(3) == approx(0.977249868, abs=1.5e-7)
    assert X.cdf(4) == approx(0.9999683287, abs=1.5e-7)
    assert X.cdf(5) == approx(0.999999999, abs=1.5e-7)
    assert X.cdf(6) == approx(1, abs=1.5e-7)
    assert X.cdf(7) == 1

    assert Y.cdf(-5) == approx(0.2458838375, abs=1.5e-7)
    assert Y.cdf(-4.32) == approx(0.3964679927, abs=1.5e-7)
    assert Y.cdf(-3.5) == approx(0.5987063257, abs=1.5e-7)
    assert Y.cdf(-2) == approx(0.8824847717, abs=1.5e-7)
    assert Y.cdf(-1.79) == approx(0.9063736475, abs=1.5e-7)
    assert Y.cdf(-0.5) == approx(0.9832066935, abs=1.5e-7)
    assert Y.cdf(1) == approx(0.9989025175, abs=1.5e-7)

    assert Z.cdf(-4) == approx(3.167126301e-5, abs=1.5e-7)
    assert Z.cdf(-3) == approx(1.349898133e-3, abs=1.5e-7)
    assert Z.cdf(-2) == approx(0.022750132, abs=1.5e-7)
    assert Z.cdf(-1) == approx(0.1586552539, abs=1.5e-7)
    assert Z.cdf(1) == approx(0.8413447461, abs=1.5e-7)
    assert Z.cdf(2) == approx(0.977249868, abs=1.5e-7)
    assert Z.cdf(3) == approx(0.9986501019, abs=1.5e-7)
    assert Z.cdf(4) == approx(0.9999683287, abs=1.5e-7)

    assert W.cdf(500) == approx(2.866524348e-7, abs=1.5e-7)
    assert W.cdf(750) == approx(6.209665428e-3, abs=1.5e-7)
    assert W.cdf(900) == approx(0.1586552539, abs=1.5e-7)
    assert W.cdf(1100) == approx(0.8413447461, abs=1.5e-7)
    assert W.cdf(1250) == approx(0.9937903346, abs=1.5e-7)
    assert W.cdf(1500) == approx(0.9999997133, abs=1.5e-7)


def test_calculate() -> None:
    """Test the use of P() to calculate probabilities of a poisson distribution."""
    X = N(2, 0.5)
    Y = N(-3.9, 1.6)
    Z = N(0, 1)

    assert P(X == 2) == approx(X.pmf(2))
    assert P(X == 12) == approx(X.pmf(12))
    assert P(X != 4) == approx(1 - X.pmf(4))
    assert P(X != 17) == approx(1 - X.pmf(17))
    assert P(X < 2) == approx(X.cdf(2))
    assert P(2 > X) == approx(X.cdf(2))
    assert P(X < 10) == approx(X.cdf(10) - X.pmf(10))
    assert P(X <= 10) == approx(X.cdf(10))
    assert P(3 < X <= 12) == approx(X.cdf(12) - X.cdf(3))
    assert P(20 >= X) == approx(X.cdf(20))
    assert P(X <= 20) == approx(X.cdf(20))
    assert P(0 <= X <= 4) == approx(X.cdf(4) - X.cdf(0) + X.pmf(0))
    assert P(7 <= X < 15) == approx(X.cdf(15) - X.cdf(7))
    assert P(3 < X < 10) == approx(X.cdf(10) - X.cdf(3))

    assert P(Y == 10) == approx(Y.pmf(10))
    assert P(Y != 6) == approx(1 - Y.pmf(6))
    assert P(Y < 10) == approx(Y.cdf(10))
    assert P(Y > 10) == approx(1 - Y.cdf(10))
    assert P(Y < 10) + P(Y > 10) == approx(1 - P(Y == 10))
    assert P(Y <= 4) == approx(Y.cdf(4))

    assert P(Z == 27) == approx(Z.pmf(27))
    assert P(Z != 33) == approx(1 - Z.pmf(33))
    assert P(Z <= 30) == approx(Z.cdf(30))
    assert P(Z > 30) == approx(1 - Z.cdf(30))
    assert P(Z >= 20) == approx(1 - Z.cdf(20))

    assert P(-0.5 < Z < 0.5) == approx(Z.cdf(0.5) - Z.cdf(-0.5))
    assert P(-1 < Z < 1) == approx(Z.cdf(1) - Z.cdf(-1))
    assert P(-1.5 < Z < 1.5) == approx(Z.cdf(1.5) - Z.cdf(-1.5))
    assert P(-2 < Z < 2) == approx(Z.cdf(2) - Z.cdf(-2))
    assert P(-2.5 < Z < 2.5) == approx(Z.cdf(2.5) - Z.cdf(-2.5))
    assert P(-3 < Z < 3) == approx(Z.cdf(3) - Z.cdf(-3))

    with pytest.raises(NonsenseError):
        P(10 < X < 8)

    with pytest.raises(NonsenseError):
        P(4 > X >= 12)

    with pytest.raises(NonsenseError):
        P(10 < X == 3)

    with pytest.raises(NonsenseError):
        P(3 == X > 10)

    with pytest.raises(NonsenseError):
        P(X == 3 > 10)

    with pytest.raises(NonsenseError):
        P(X == 3 < 10)

    with pytest.raises(NonsenseError):
        P(3 != X > 10)

    with pytest.raises(NonsenseError):
        P(X != 3 > 10)
