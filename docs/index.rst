Hello, sp.fem!
==============

This is the documentation of a simple finite element assembler library written in Python 2.7. The library is useful for quickly creating numerical solvers for various PDE-based models.

The library is currently work-in-progress and there is a fair amount of work to be done until it can be considered complete. The current state is however more than usable.

What the hell is a finite element?
==================================

You might remember computing derivatives in high school. Unfortunately, back then the practice was not motivated that much (same applies to many university-level mathematics courses) and probably you ended up forgetting most of what you learned.

This is unfortunate since the derivatives are used to build very useful and detailed models of physics, engineering and economics. These models are known as *differential equations* and very often can be robustly solved using a computer and the so-called *finite element method*.

This library contains tools to quickly build the matrices related to the finite element method. We support various types of finite elements out-of-the-box although defining completely new elements is also possible.

Getting started
===============

You can download the library and get started by running the following commands

.. code-block:: bash

    git clone https://github.com/kinnala/sp.fem
    cd sp.fem

If you are a well-seasoned Python developer you may look into the contents of requirements.txt, check that you have all the required libraries and do whatever you wish.

Otherwise, we suggest that you use *miniconda* for managing Python virtual environments and installing packages. You can download and install *miniconda* by running

.. code-block:: bash

    make install-conda

Next you can create a new virtual environment and install the required packages by running

.. code-block:: bash

    make dev-install

The newly created virtual environment can be activated by writing

.. code-block:: bash

    source activate spfemenv

Optionally, in order to use the geometry module, you should install MeshPy dependency by running

.. code-block:: bash

    pip install meshpy

Tutorial
========

Derivative describes the change. For example, the rate of change in the location of a particle is called the particle's velocity. In technical terms we say that the derivative of the location (with respect to time) is the velocity. Similarly, the derivative of the velocity is called acceleration.

One can consider an infinite lattice of point masses and springs. Let :math:`u(x,y,t)` be. The movement of the points is described by the so-called wave equation which states that the acceleration of a point is proportional to the curvature, that is,

.. math::

    u_{tt}=\Delta u

Classes
=======

This section contains documentation generated automatically from the source code of the relevant classes.

fem.mesh
########

.. automodule:: spfem.mesh
    :members:

fem.asm
#######

.. automodule:: spfem.asm
    :members:

fem.element
###########

.. automodule:: spfem.element
    :members:

fem.mapping
###########

.. automodule:: spfem.mapping
    :members:

fem.utils
#########

.. automodule:: spfem.utils
    :members:

Tips
====

* Errors related to qt4, wx, mayavi, etc. can be sometimes fixed by simply changing environment variables or running ipython with the following flags:

.. code-block:: bash

    ipython --gui=wx --pylab=wx

.. code-block:: bash

    ETS_TOOLKIT=qt4 ipython --gui=wx --pylab=wx

* Simplest way to run tests is to discover them all using unittest as follows:

.. code-block:: bash

    ipython -m unittest discover ./spfem

* In order to estimate test coverage you can install coverage.py and run it

.. code-block:: bash

    pip install coverage
    coverage run -m unittest discover ./spfem
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

