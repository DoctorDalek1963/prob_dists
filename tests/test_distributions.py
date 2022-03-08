# prob_dists - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""A simple test module to test the :mod:`prob_dists.distributions` module.

All test values calculated with Casio fx-991ES.
"""
import pytest
from pytest import approx

import prob_dists as pd


def test_binomial_pmf() -> None:
    """Test the binomial distribution probability mass function."""
    X = pd.B(20, 0.25)
    Y = pd.B(14, 0.36)
    Z = pd.B(50, 0.782)

    assert X.probability_mass_function(0) == approx(3.171211939e-3)
    assert X.probability_mass_function(1) == approx(0.02114141293)
    assert X.probability_mass_function(2) == approx(0.0669478076)
    assert X.probability_mass_function(3) == approx(0.1338956152)
    assert X.probability_mass_function(4) == approx(0.1896854549)
    assert X.probability_mass_function(5) == approx(0.2023311519)
    assert X.probability_mass_function(6) == approx(0.1686092932)
    assert X.probability_mass_function(7) == approx(0.1124061955)
    assert X.probability_mass_function(8) == approx(0.06088668922)
    assert X.probability_mass_function(9) == approx(0.02706075076)
    assert X.probability_mass_function(10) == approx(9.92227528e-3)
    assert X.probability_mass_function(11) == approx(3.006750085e-3)
    assert X.probability_mass_function(12) == approx(7.516875212e-4)
    assert X.probability_mass_function(13) == approx(1.54192312e-4)
    assert X.probability_mass_function(14) == approx(2.569871867e-5)
    assert X.probability_mass_function(15) == approx(3.426495823e-6)
    assert X.probability_mass_function(16) == approx(3.569266482e-7)
    assert X.probability_mass_function(17) == approx(2.799424692e-8)
    assert X.probability_mass_function(18) == approx(1.55523594e-9)
    assert X.probability_mass_function(19) == approx(5.456968211e-11)
    assert X.probability_mass_function(20) == approx(9.094947018e-13)

    assert Y.probability_mass_function(4) == approx(0.1938401871)
    assert Y.probability_mass_function(8) == approx(0.05821771856)
    assert Y.probability_mass_function(13) == approx(1.528412284e-5)

    assert Z.probability_mass_function(12) == approx(4.613656329e-16)
    assert Z.probability_mass_function(21) == approx(2.516357132e-8)
    assert Z.probability_mass_function(36) == approx(0.0734455353)
    assert Z.probability_mass_function(45) == approx(0.01631804512)

    with pytest.raises(pd.NonsenseError):
        X.probability_mass_function(-1, strict=True)
        X.probability_mass_function(-3, strict=True)
        X.probability_mass_function(21, strict=True)
        X.probability_mass_function(30, strict=True)

        X.probability_mass_function(12.5, strict=True)  # type: ignore[arg-type]
        X.probability_mass_function(22.3, strict=True)  # type: ignore[arg-type]

    assert X.probability_mass_function(-1, strict=False) == 0
    assert X.probability_mass_function(-3, strict=False) == 0
    assert X.probability_mass_function(21, strict=False) == 0
    assert X.probability_mass_function(30, strict=False) == 0

    assert X.probability_mass_function(12.5, strict=False) == 0  # type: ignore[arg-type]
    assert X.probability_mass_function(22.3, strict=False) == 0  # type: ignore[arg-type]
