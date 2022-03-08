# prob_dists - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""This module contains classes for various probability distributions."""

import abc
from typing import Literal

from .utility import choose


class NonsenseError(Exception):
    """A simple error representing mathematical nonsense.

    This could be a probability that doesn't make sense, or getting more successes than trials, etc.
    """


class Distribution(metaclass=abc.ABCMeta):
    """This is an abstract superclass representing an arbitrary probability distribution.

    It has abstract methods which must be implemented by any concrete subclasses.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """Create a Distribution object."""

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return a nice, readable repr of the Distribution."""

    def __str__(self) -> str:
        """Return the repr by default."""
        return repr(self)

    @abc.abstractmethod
    def probability_mass_function(self, value: int, *, strict: bool = True) -> float:
        """Evaluate the PMF of this distribution.

        This is the probability that a random variable distributed by this
        distribution takes on the given value.

        :param int value: The value to find the probability of
        :param bool strict: Whether to throw errors for invalid input, or return 0
        :return float: The calculated probability

        :raises NonsenseError: If the value doesn't make sense in the context of the distribution
        """

    @abc.abstractmethod
    def cumulative_distribution_function(self, value: int, *, strict: bool = True) -> float:
        """Evaluate the CDF of this distribution.

        This is the probability that a random variable distributed by this
        distribution takes on a value less than or equal to the given value.

        :param int value: The value to find the probability of
        :param bool strict: Whether to throw errors for invalid input, or return 0
        :return float: The calculated probability

        :raises NonsenseError: If the value doesn't make sense in the context of the distribution
        """


class BinomialDistribution(Distribution):
    """This is a binomial distribution, used to model multiple independent, binary trials."""

    def __init__(self, number_of_trials: int, probability: float):
        """Construct a binomial distribution from a given number of trials and probability of success."""
        if not 0 <= probability <= 1:
            raise NonsenseError('Binomial probability must be between 0 and 1')

        self._number_of_trials = number_of_trials
        self._probability = probability

    def __repr__(self) -> str:
        """Return a nice repr of the distribution."""
        return f'B({self._number_of_trials}, {self._probability})'

    def _choose(self, r: int) -> int:
        """Call :meth:`prob_dists.utility.choose` with the number of trials and the given value."""
        return choose(self._number_of_trials, r)

    def _check_nonsense(self, successes: int, strict: bool) -> Literal[None, -1]:
        """Check if this number of successes is nonsense.

        :param int successes: The number of successes to check
        :param bool strict: Whether to throw errors or just return 0
        :returns: None on success, -1 on fail
        :rtype: Literal[None, -1]

        :raises NonsenseError: If the number of successes is more than the number of trials
        :raises NonsenseError: If the number of successes is not an integer
        """
        if successes < 0:
            if strict:
                raise NonsenseError(f'Cannot have negative number of successes ({successes})')

            return -1

        if successes > self._number_of_trials:
            if strict:
                raise NonsenseError(f'Cannot have more success ({successes}) than trials ({self._number_of_trials})')

            return -1

        if successes != int(successes):
            if strict:
                raise NonsenseError(f'Cannot ask probability of {successes} successes')

            return -1

        return None

    def probability_mass_function(self, successes: int, *, strict: bool = True) -> float:
        """Return the probability that we get a given number of successes.

        :param int successes: The number of successes to find the probability of
        :param bool strict: Whether to throw errors for invalid input, or return 0
        :return float: The probability of getting exactly this many successes

        :raises NonsenseError: If the number of successes is more than the number of trials
        :raises NonsenseError: If the number of successes is not an integer
        """
        return 0 if self._check_nonsense(successes, strict) is not None else \
            self._choose(successes) * (self._probability ** successes) * \
            ((1 - self._probability) ** (self._number_of_trials - successes))

    def cumulative_distribution_function(self, successes: int, *, strict: bool = True) -> float:
        """Return the probability that we got less than or equal to the given number of successes.

        :param int successes: The number of successes to find the probability of
        :param bool strict: Whether to throw errors for invalid input, or return 0
        :return float: The probability of getting exactly this many successes

        :raises NonsenseError: If the number of successes is more than the number of trials
        :raises NonsenseError: If the number of successes is not an integer
        """
        # mypy expects this sum to have ints for some reason, so we ignore it
        return 0 if self._check_nonsense(successes, strict) is not None else \
            sum(self.probability_mass_function(x) for x in range(successes + 1))  # type: ignore[misc]
