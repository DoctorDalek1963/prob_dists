# prob_dists - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""This is the top-level ``prob_dists`` package, which contains all the subpackages of the project."""

from . import distributions
from .distributions import NonsenseError

B = distributions.BinomialDistribution

__all__ = ['B', 'distributions', 'NonsenseError']

__version__ = '0.0.1-alpha'
