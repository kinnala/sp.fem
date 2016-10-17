Hello, sp.fem!
==============

This is the documentation of a simple finite element assembler library written in Python 2.7. The library is useful for quickly creating numerical solvers for various PDE-based models.

The library is currently work-in-progress and there is a fair amount of work to be done until it can be considered complete. The current state is however more than usable.

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

This tutorial is targeted towards people who have a basic understanding what partial differential equations are and want to start solving them using finite elements.

Let us start by solving the wave equation,

.. math::

    u_{tt}=\Delta u,

in a two-dimensional domain. The time discretization is performed using finite differences and spatial discretization using finite elements in a classical manner.

Denote the domain by :math:`\Omega` and the end time by :math:`T`.
The weak formulation reads: for all :math:`t \in [0,T]`, find :math:`u(t) \in H^1_0(\Omega)` satisfying

.. math::
    (u_{tt},v)+(\nabla u, \nabla v)=0

for every :math:`v \in H^1_0(\Omega)`.

Let

.. math::
    
    u_h(x,t)=\sum_{j=1}^N u_j(t) \varphi_j(x).

The spatial discretization leads to

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

