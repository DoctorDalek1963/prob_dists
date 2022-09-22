# probcalc - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""A simple test module to test the :class:`probcalc.distributions.GeometricDistribution`.

All test values calculated with a Casio fx-991EX Classwiz.
"""

import pytest
from pytest import approx

from probcalc import P, Geo, NonsenseError


def test_pmf() -> None:
    """Test the geometric distribution PMF."""
    X = Geo(0.2)
    Y = Geo(0.56)
    Z = Geo(0.34521)

    assert X.pmf(1) == approx(0.2)
    assert X.pmf(2) == approx(0.16)
    assert X.pmf(3) == approx(0.128)
    assert X.pmf(4) == approx(0.1024)
    assert X.pmf(5) == approx(0.08192)
    assert X.pmf(6) == approx(0.065536)
    assert X.pmf(7) == approx(0.0524288)
    assert X.pmf(8) == approx(0.04194304)
    assert X.pmf(9) == approx(0.033554432)
    assert X.pmf(10) == approx(0.0268435456)
    assert X.pmf(11) == approx(0.02147483648)
    assert X.pmf(12) == approx(0.01717986918)
    assert X.pmf(13) == approx(0.01374389535)
    assert X.pmf(14) == approx(0.01099511628)
    assert X.pmf(15) == approx(0.008796093022)
    assert X.pmf(16) == approx(0.007036874418)
    assert X.pmf(17) == approx(0.005629499534)
    assert X.pmf(18) == approx(0.004503599627)
    assert X.pmf(19) == approx(0.003602879702)
    assert X.pmf(20) == approx(0.002882303762)

    assert Y.pmf(4) == approx(0.04770304)
    assert Y.pmf(8) == approx(0.001787955734)
    assert Y.pmf(13) == approx(2.948629083e-5)

    assert Z.pmf(12) == approx(0.003274958403)
    assert Z.pmf(21) == approx(7.246430514e-5)
    assert Z.pmf(36) == approx(1.263731193e-7)
    assert Z.pmf(45) == approx(2.79623102e-9)

    for num in [0, -1, -3, 12.5, 22.3, -2.41, 916.02]:
        with pytest.raises(NonsenseError):
            X.pmf(num, strict=True)  # type: ignore[arg-type]

    assert X.pmf(0, strict=False) < 1e-100
    assert X.pmf(-1, strict=False) < 1e-100
    assert X.pmf(-3, strict=False) < 1e-100
    assert X.pmf(12.5, strict=False) < 1e-100  # type: ignore[arg-type]
    assert X.pmf(22.3, strict=False) < 1e-100  # type: ignore[arg-type]
    assert X.pmf(-2.41, strict=False) < 1e-100  # type: ignore[arg-type]
    assert X.pmf(916.02, strict=False) < 1e-100  # type: ignore[arg-type]


def test_cdf() -> None:
    """Test the geometric distribution CDF."""
    X = Geo(0.2)
    Y = Geo(0.56)
    Z = Geo(0.34521)

    assert X.cdf(1) == approx(0.2)
    assert X.cdf(2) == approx(0.36)
    assert X.cdf(3) == approx(0.488)
    assert X.cdf(4) == approx(0.5904)
    assert X.cdf(5) == approx(0.67232)
    assert X.cdf(6) == approx(0.737856)
    assert X.cdf(7) == approx(0.7902848)
    assert X.cdf(8) == approx(0.83222784)
    assert X.cdf(9) == approx(0.865782272)
    assert X.cdf(10) == approx(0.8926258176)
    assert X.cdf(11) == approx(0.9141006541)
    assert X.cdf(12) == approx(0.9312805233)
    assert X.cdf(13) == approx(0.9450244186)
    assert X.cdf(14) == approx(0.9560195349)
    assert X.cdf(15) == approx(0.9648156279)
    assert X.cdf(16) == approx(0.9718525023)
    assert X.cdf(17) == approx(0.9774820019)
    assert X.cdf(18) == approx(0.9819856015)
    assert X.cdf(19) == approx(0.9855884812)
    assert X.cdf(20) == approx(0.988470785)

    assert Y.cdf(4) == approx(0.96251904)
    assert Y.cdf(8) == approx(0.9985951776)
    assert Y.cdf(13) == approx(0.9999768322)

    assert Z.cdf(12) == approx(0.9937881)
    assert Z.cdf(21) == approx(0.9998625506)
    assert Z.cdf(36) == approx(0.9999997603)
    assert Z.cdf(45) == approx(0.9999999947)

    for num in [0, -1, -3, 12.5, 22.3, -2.41, 916.02]:
        with pytest.raises(NonsenseError):
            X.cdf(num, strict=True)  # type: ignore[arg-type]

    assert X.cdf(0, strict=False) < 1e-100
    assert X.cdf(-1, strict=False) < 1e-100
    assert X.cdf(-3, strict=False) < 1e-100
    assert X.cdf(12.5, strict=False) < 1e-100  # type: ignore[arg-type]
    assert X.cdf(22.3, strict=False) < 1e-100  # type: ignore[arg-type]
    assert X.cdf(-2.41, strict=False) < 1e-100  # type: ignore[arg-type]
    assert X.cdf(916.02, strict=False) < 1e-100  # type: ignore[arg-type]


def test_calculate() -> None:
    """Test the use of P() to calculate probabilities of a geometric distribution."""
    X = Geo(0.2)
    Y = Geo(0.56)
    Z = Geo(0.34521)

    assert P(X == 2) == approx(X.pmf(2))
    assert P(X == 12) == approx(X.pmf(12))
    assert P(X != 4) == approx(1 - X.pmf(4))
    assert P(X != 17) == approx(1 - X.pmf(17))
    assert P(X < 2) == approx(X.cdf(1))
    assert P(2 > X) == approx(X.cdf(1))
    assert P(X < 10) == approx(X.cdf(9))
    assert P(X <= 10) == approx(X.cdf(10))
    assert P(3 < X <= 12) == approx(sum(X.pmf(x) for x in range(4, 13)))
    assert P(7 <= X < 15) == approx(sum(X.pmf(x) for x in range(7, 15)))
    assert P(3 < X < 10) == approx(sum(X.pmf(x) for x in range(4, 10)))

    assert P(Y == 10) == approx(Y.pmf(10))
    assert P(Y != 6) == approx(1 - Y.pmf(6))
    assert P(Y < 10) == approx(Y.cdf(9))
    assert P(Y > 10) == approx(1 - Y.cdf(10))
    assert P(Y < 10) + P(Y > 10) == approx(1 - P(Y == 10))
    assert P(Y <= 4) == approx(Y.cdf(4))

    assert P(Z == 27) == approx(Z.pmf(27))
    assert P(Z != 33) == approx(1 - Z.pmf(33))
    assert P(Z <= 30) == approx(Z.cdf(30))
    assert P(Z > 30) == approx(1 - Z.cdf(30))
    assert P(Z >= 20) == approx(1 - Z.cdf(19))

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
