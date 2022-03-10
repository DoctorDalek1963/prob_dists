# prob_dists - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""This is the top-level ``prob_dists`` package, which contains all the subpackages and submodules of the project.

Here's a table of user-friendly alises and the backend classes they refer to:

.. list-table::
   :widths: 20 70
   :header-rows: 1

   * - Alias
     - Class or function name
   * - P
     - :func:`prob_dists.distributions.calculate_probability`
   * - B
     - :class:`prob_dists.distributions.BinomialDistribution`
   * - Po
     - :class:`prob_dists.distributions.PoissonDistribution`
"""

from . import distributions, utility
from .distributions import NonsenseError

P = distributions.calculate_probability
B = distributions.BinomialDistribution
Po = distributions.PoissonDistribution

__all__ = ['P', 'B', 'Po', 'NonsenseError', 'distributions', 'utility']

__version__ = '0.1.2'
