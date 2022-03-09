Welcome to the prob_dists docs
==============================

This is the documentation for prob_dists version |release|.

This is a project that aims to make calculations with random variables
distributed by various probability distributions as easy as possible.

Here's a simple example of the kind of things that this project lets you do:

:Example:

>>> from prob_dists import P, B
>>> X = B(30, 0.25)
>>> P(X > 12)
0.02159364088
>>> P(10 <= X < 20)
0.1965915424

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   prob_dists

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
