Welcome to the probcalc docs
============================

.. _pylint: https://pylint.org/

This is the documentation for probcalc version |release|.

This is a project that aims to make calculations with random variables
distributed by various probability distributions as easy as possible.

Here's a simple example of the kind of things that this project lets you do:

:Example:

>>> from probcalc import P, B
>>> X = B(30, 0.25)
>>> P(X > 12)
0.02159364088
>>> P(10 <= X < 20)
0.1965915424

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   probcalc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
