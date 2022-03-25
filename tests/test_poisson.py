# probcalc - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""A simple test module to test the :class:`probcalc.distributions.PoissonDistributions`.

All test values calculated with a Casio fx-991EX Classwiz.
"""

import pytest
from pytest import approx

from probcalc import P, Po, NonsenseError


def test_pmf() -> None:
    """Test the Poisson distribution PMF."""
    X = Po(2)
    Y = Po(12.3)
    Z = Po(8.362)
    W = Po(1000)

    assert X.pmf(0) == approx(0.1353352832)
    assert X.pmf(1) == approx(0.2706705665)
    assert X.pmf(2) == approx(0.2706705665)
    assert X.pmf(3) == approx(0.1804470443)
    assert X.pmf(4) == approx(0.09022352216)
    assert X.pmf(5) == approx(0.03608940886)
    assert X.pmf(10) == approx(3.818985065e-5)
    assert X.pmf(20) == approx(5.832924198e-14)

    assert Y.pmf(0) == approx(4.551744463e-6)
    assert Y.pmf(1) == approx(5.59864569e-5)
    assert Y.pmf(2) == approx(3.443167099e-4)
    assert Y.pmf(3) == approx(1.411698511e-3)
    assert Y.pmf(4) == approx(4.34097292e-3)
    assert Y.pmf(5) == approx(0.01067879338)
    assert Y.pmf(10) == approx(0.09941821344)
    assert Y.pmf(20) == approx(0.01175318263)

    assert Z.pmf(0) == approx(2.335767129e-4)
    assert Z.pmf(1) == approx(1.953168473e-3)
    assert Z.pmf(2) == approx(8.166197386e-3)
    assert Z.pmf(3) == approx(0.02276191418)
    assert Z.pmf(4) == approx(0.04758378159)
    assert Z.pmf(5) == approx(0.07957911634)
    assert Z.pmf(10) == approx(0.1075890671)
    assert Z.pmf(20) == approx(2.682305189e-4)

    assert W.pmf(100) < 1e-100
    assert W.pmf(200) < 1e-100
    assert W.pmf(300) < 1e-100
    assert W.pmf(400) < 1e-100
    assert W.pmf(500) == approx(4.160155476e-69)
    assert W.pmf(600) == approx(4.010801147e-43)
    assert W.pmf(700) == approx(2.095736914e-24)
    assert W.pmf(800) == approx(6.583151642e-12)
    assert W.pmf(900) == approx(7.516954352e-5)

    with pytest.raises(NonsenseError):
        X.pmf(-1)

    with pytest.raises(NonsenseError):
        X.pmf(2.3)  # type: ignore[arg-type]


def test_cdf() -> None:
    """Test the Poisson distribution CDF."""
    X = Po(2)
    Y = Po(12.3)
    Z = Po(8.362)
    W = Po(1000)

    assert X.cdf(0) == approx(0.1353352832)
    assert X.cdf(1) == approx(0.406005879)
    assert X.cdf(2) == approx(0.6766764201)
    assert X.cdf(3) == approx(0.8571234644)
    assert X.cdf(4) == approx(0.9473469831)
    assert X.cdf(5) == approx(0.983436392)
    assert X.cdf(10) == approx(0.9999916918)
    assert X.cdf(20) == approx(1)

    assert Y.cdf(0) == approx(4.551744463e-6)
    assert Y.cdf(1) == approx(6.053820136e-5)
    assert Y.cdf(2) == approx(4.048549113e-4)
    assert Y.cdf(3) == approx(1.816553422e-3)
    assert Y.cdf(4) == approx(6.157526342e-3)
    assert Y.cdf(5) == approx(0.01683631973)
    assert Y.cdf(10) == approx(0.3165827183)
    assert Y.cdf(20) == approx(0.9851937561)

    assert Z.cdf(0) == approx(2.335767129e-4)
    assert Z.cdf(1) == approx(2.186745186e-3)
    assert Z.cdf(2) == approx(0.01035294257)
    assert Z.cdf(3) == approx(0.03311485675)
    assert Z.cdf(4) == approx(0.08069863835)
    assert Z.cdf(5) == approx(0.1602777547)
    assert Z.cdf(10) == approx(0.7784049707)
    assert Z.cdf(20) == approx(0.999830179)

    assert W.cdf(100) < 1e-100
    assert W.cdf(200) < 1e-100
    assert W.cdf(300) < 1e-100
    assert W.cdf(400) < 1e-100
    assert W.cdf(500) == approx(8.303834064e-69)
    assert W.cdf(600) == approx(9.989996782e-43)
    assert W.cdf(700) == approx(6.933009918e-24)
    assert W.cdf(800) == approx(3.229888671e-11)
    assert W.cdf(900) == approx(6.97767356e-4)

    with pytest.raises(NonsenseError):
        X.cdf(-1)

    with pytest.raises(NonsenseError):
        X.cdf(2.3)  # type: ignore[arg-type]


def test_calculate() -> None:
    """Test the use of P() to calculate probabilities of a poisson distribution."""
    X = Po(2)
    Y = Po(12.3)
    Z = Po(8.362)

    # This is a carbon copy of the binomial version of this test but I'm lazy
    # I can't factor this out easily because the distributions are slightly different
    assert P(X == 2) == approx(X.pmf(2))
    assert P(X == 12) == approx(X.pmf(12))
    assert P(X != 4) == approx(1 - X.pmf(4))
    assert P(X != 17) == approx(1 - X.pmf(17))
    assert P(X < 2) == approx(sum(X.pmf(x) for x in (0, 1)))
    assert P(2 > X) == approx(sum(X.pmf(x) for x in (0, 1)))
    assert P(X < 10) == approx(sum(X.pmf(x) for x in range(10)))
    assert P(X <= 10) == approx(sum(X.pmf(x) for x in range(11)))
    assert P(3 < X <= 12) == approx(sum(X.pmf(x) for x in range(4, 13)))
    assert P(20 >= X) == 1
    assert P(X <= 20) == 1
    assert P(0 <= X <= 20) == 1
    assert P(7 <= X < 15) == approx(sum(X.pmf(x) for x in range(7, 15)))
    assert P(3 < X < 10) == approx(sum(X.pmf(x) for x in range(4, 10)))

    assert P(Y == 10) == approx(Y.pmf(10))
    assert P(Y != 6) == approx(1 - Y.pmf(6))
    assert P(Y < 10) == approx(sum(Y.pmf(x) for x in range(10)))
    assert P(Y > 10) == approx(1 - sum(Y.pmf(x) for x in range(11)))  # type: ignore[misc]
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
