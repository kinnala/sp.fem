Hello, sp.fem!
==============

This is the documentation of a simple finite element assembler library written in Python 2.7. The library is useful for quickly creating numerical solvers for various PDE-based models.

What the hell is a finite element?
==================================

You might remember computing derivatives in high school. Unfortunately, back then the practice was not motivated that much (same applies to many university-level mathematics courses) and probably you ended up forgetting most of what you learned.

This is unfortunate since the derivatives are used to build very useful and detailed models of physics, engineering and economics. These models are known as *differential equations* and very often can be robustly solved using a computer and the so-called *finite element method*.

Getting started
===============

installation instructions

Tutorial
========

first usage example

Classes
=======

This section contains documentation generated automatically from the source code of the relevant classes.

fem.mesh
########

.. automodule:: fem.mesh
    :members:

fem.asm
#######

.. automodule:: fem.asm
    :members:

fem.element
###########

.. automodule:: fem.element
    :members:

fem.mapping
###########

.. automodule:: fem.mapping
    :members:

Tips
====

* Errors related to qt4, wx, mayavi, etc. can be sometimes fixed by simply changing environment variables or running ipython with the following flags:

.. code-block:: none

    ipython --gui=wx --pylab=wx

.. code-block:: none

    ETS_TOOLKIT=qt4 ipython --gui=wx --pylab=wx

* Simplest way to run tests is to discover them all using unittest as follows:

.. code-block:: none

    ipython -m unittest discover ./fem

* In order to estimate test coverage you can install coverage.py and run it

.. code-block:: none

    pip install coverage
    coverage run -m unittest discover ./fem
    coverage html

License
=======

sp.fem is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

sp.fem is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with sp.fem. If not, see <http://www.gnu.org/licenses/>.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

