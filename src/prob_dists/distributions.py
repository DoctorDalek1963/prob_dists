# prob_dists - Calculate probabilities for distributions
# Copyright (C) 2022 D. Dyson (DoctorDalek1963)

# This program is licensed under GNU GPLv3, available here:
# <https://www.gnu.org/licenses/gpl-3.0.html>

"""This module contains classes for various probability distributions."""

import abc
from typing import Literal

from .utility import choose, round_sig_fig


class NonsenseError(Exception):
    """A simple error representing mathematical nonsense.

    This could be a probability that doesn't make sense, or getting more successes than trials, etc.
    """


class _Bounds:
    """This is a simple little class to hold bounds for a distribution."""

    def __init__(self):
        """Create a :class:`_Bounds` object with default bounds."""
        self.lower: tuple[int | None, bool] = (None, False)
        self.upper: tuple[int | None, bool] = (None, False)

    def __repr__(self) -> str:
        """Return a simple repr of the object."""
        return f'{self.__class__.__module__}.{self.__class__.__name__}({self.lower}, {self.upper})'

    def __eq__(self, other):
        """Check equality."""
        if not isinstance(other, _Bounds):
            return NotImplemented

        return self.lower == other.lower and self.upper == other.upper


class Distribution(metaclass=abc.ABCMeta):
    """This is an abstract superclass representing an arbitrary probability distribution.

    It has abstract methods which must be implemented by any concrete subclasses.
    """

    def __init__(self, *args, **kwargs):
        """Create a Distribution object."""
        # This tuple represents the lower and upper bounds of the distribution
        # The bool is whether the number itself should be included in the calculation
        self._bounds = _Bounds()
        self._accepts_floats: bool = False

    def reset_bounds(self) -> None:
        """Reset the bounds of the distribution."""
        self._bounds = _Bounds()

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Return a nice, readable repr of the Distribution."""

    def __str__(self) -> str:
        """Return the repr by default."""
        return repr(self)

    def __eq__(self, other):
        """Set the upper and lower bounds to ``other``."""
        if not (isinstance(other, int) or (self._accepts_floats and isinstance(other, float))):
            return NotImplemented

        # If the bounds are already mutated, then we've mixed inequality and equality
        if self._bounds != _Bounds():
            raise NonsenseError('Cannot have inequality and equality mixed together')

        self._bounds.upper = (other, True)
        self._bounds.lower = (other, True)
        return self

    def __lt__(self, other):
        """Set the upper bound accordingly."""
        if not (isinstance(other, int) or (self._accepts_floats and isinstance(other, float))):
            return NotImplemented

        self._bounds.upper = (other, False)
        return self

    def __le__(self, other):
        """Set the upper bound accordingly."""
        if not (isinstance(other, int) or (self._accepts_floats and isinstance(other, float))):
            return NotImplemented

        self._bounds.upper = (other, True)
        return self

    def __gt__(self, other):
        """Set the lower bound accordingly."""
        if not (isinstance(other, int) or (self._accepts_floats and isinstance(other, float))):
            return NotImplemented

        self._bounds.lower = (other, False)
        return self

    def __ge__(self, other):
        """Set the lower bound accordingly."""
        if not (isinstance(other, int) or (self._accepts_floats and isinstance(other, float))):
            return NotImplemented

        self._bounds.lower = (other, True)
        return self

    def calculate(self, *, strict: bool = True) -> float:
        """Return the probability of a random variable distributed like this taking on a value within its bounds."""
        lower = self._bounds.lower
        upper = self._bounds.upper

        probability = 1.0

        if upper[0] is not None:
            probability = self.cdf(upper[0], strict=strict)

            if not upper[1]:
                probability -= self.pmf(upper[0], strict=strict)

        if lower[0] is not None:
            probability -= self.cdf(lower[0], strict=strict)

            if lower[1]:
                probability += self.pmf(lower[0], strict=strict)

        if probability < 0:
            raise NonsenseError('This inequality doesn\'t make sense')

        return round_sig_fig(probability, 10)

    @abc.abstractmethod
    def pmf(self, value: int, *, strict: bool = True) -> float:
        """Evaluate the PMF of this distribution.

        This is the probability that a random variable distributed by this
        distribution takes on the given value.

        :param int value: The value to find the probability of
        :param bool strict: Whether to throw errors for invalid input, or return 0
        :return float: The calculated probability

        :raises NonsenseError: If the value doesn't make sense in the context of the distribution
        """

    @abc.abstractmethod
    def cdf(self, value: int, *, strict: bool = True) -> float:
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

        super().__init__()

        self._accepts_floats = False

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

    def pmf(self, successes: int, *, strict: bool = True) -> float:
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

    def cdf(self, successes: int, *, strict: bool = True) -> float:
        """Return the probability that we got less than or equal to the given number of successes.

        :param int successes: The number of successes to find the probability of
        :param bool strict: Whether to throw errors for invalid input, or return 0
        :return float: The probability of getting exactly this many successes

        :raises NonsenseError: If the number of successes is more than the number of trials
        :raises NonsenseError: If the number of successes is not an integer
        """
        # mypy expects this sum to have ints for some reason, so we ignore it
        return 0 if self._check_nonsense(successes, strict) is not None else \
            sum(self.pmf(x) for x in range(successes + 1))  # type: ignore[misc]


def calculate_probability(distribution: Distribution) -> float:
    """Return the calculated probability of a random variable for this distribution getting within its bounds.

    This function gets exported as ``P`` by ``__init__.py``, which lets the user do things like:
    >>> from prob_dists import P, B
    >>> X = B(20, 0.5)
    >>> P(X > 6)
    0.9423408508
    >>> P(4 < X <= 12)
    0.8625030518
    """
    probability = distribution.calculate(strict=True)
    distribution.reset_bounds()
    return probability
