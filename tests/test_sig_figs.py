# probcalc - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""A simple test module to test the :meth:`probcalc.distribution_classes.ProbabilityCalculator.set_sig_figs` method.

All test values calculated with a Casio fx-991EX Classwiz.
"""

from probcalc import P, B, Po, N


def test_sig_figs() -> None:
    """Test sig figs for the binomial distribution."""
    X = B(20, 0.25)
    Y = Po(12.3)
    Z = N(-3.9, 1.6)

    assert str(P(X > 10)) == '0.003942141664'
    assert str(P(X < 5)) == '0.4148415025'
    assert str(P(2 <= X < 6)) == '0.5928600295'

    assert str(P(Y > 10)) == '0.6834172813'
    assert str(P(Y < 5)) == '0.006157526342'
    assert str(P(2 <= Y < 6)) == '0.01677578152'

    assert str(P(Z > 10)) == '0.0'
    assert str(P(Z < 5)) == '0.9999999867'
    assert str(P(2 <= Z < 6)) == '0.0001132337412'

    P.set_sig_figs(6)

    assert str(P(X > 10)) == '0.00394214'
    assert str(P(X < 5)) == '0.414842'
    assert str(P(2 <= X < 6)) == '0.59286'

    assert str(P(Y > 10)) == '0.683417'
    assert str(P(Y < 5)) == '0.00615753'
    assert str(P(2 <= Y < 6)) == '0.0167758'

    assert str(P(Z > 10)) == '0.0'
    assert str(P(Z < 5)) == '1.0'
    assert str(P(2 <= Z < 6)) == '0.000113234'

    P.set_sig_figs(4)

    assert str(P(X > 10)) == '0.003942'
    assert str(P(X < 5)) == '0.4148'
    assert str(P(2 <= X < 6)) == '0.5929'

    assert str(P(Y > 10)) == '0.6834'
    assert str(P(Y < 5)) == '0.006158'
    assert str(P(2 <= Y < 6)) == '0.01678'

    assert str(P(Z > 10)) == '0.0'
    assert str(P(Z < 5)) == '1.0'
    assert str(P(2 <= Z < 6)) == '0.0001132'
