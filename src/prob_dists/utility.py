# prob_dists - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""A simple utility module to provide helper functions for the maths."""

from functools import reduce
from operator import mul


def factorial(n: int) -> int:
    """Return the factorial of ``n``."""
    return reduce(mul, range(1, n + 1), 1)


def choose(n: int, r: int) -> int:
    """Return the number of ways to choose ``r`` items from ``n`` elements.

    :param int n: The number of items to choose from
    :param int r: The number of items to be chosen
    :return int: The number of ways to choose ``r`` from ``n``

    :raises ValueError: If ``r > n``
    """
    if r > n:
        raise ValueError(f'Cannot choose {r} items from only {n} elements')

    return factorial(n) // (factorial(r) * factorial(n - r))
