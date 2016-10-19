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

We begin by importing the necessary library functions.

.. code-block:: python

    from spfem.mesh import MeshTri
    from spfem.asm import AssemblerElement
    from spfem.element import ElementTriP1
    from spfem.utils import direct

Let us solve the Poisson equation in a unit square :math:`\Omega = [0,1]^2` with unit loading. We can obtain a mesh of the unit square and refine it six times by

.. code-block:: python

    m=MeshTri()
    m.refine(6)

By default, the initializer of :class:`spfem.mesh.MeshTri` returns a mesh of the unit square with two elements. The :py:meth:`spfem.mesh.MeshTri.refine` method refines the mesh by splitting each triangle into four subtriangles. Let us denote the finite element mesh by :math:`\mathcal{T}_h` and an arbitrary element by :math:`K`.

The governing equation is

.. math::

    -\Delta u=1,

and it is combined with the boundary condition

.. math::
    
    u=0.

The weak formulation reads: find :math:`u \in H^1_0(\Omega)` satisfying

.. math::

    (\nabla u, \nabla v)=(1,v)

for every :math:`v \in H^1_0(\Omega)`.

We use a conforming piecewise linear finite element approximation space

.. math::

    V_h = \{ w_h \in H^1_0(\Omega) : w_h|_K \in P_1(K)~\forall K \in \mathcal{T}_h \}.

The finite element method reads: find :math:`u_h \in V_h` satisfying

.. math::
    
    (\nabla u_h, \nabla v_h) = (1,v_h)

for every :math:`v_h \in V_h`. A typical approach to impose the boundary condition :math:`u_h=0` (which is implicitly included in the definition of :math:`V_h`) is to initially build the matrix and the vector corresponding to the discrete space

.. math::
    
    W_h = \{ w_h \in H^1(\Omega) : w_h|_K \in P_1(K)~\forall K \in \mathcal{T}_h \}

and afterwards remove the rows and columns corresponding to the boundary nodes. An assembler object corresponding to the mesh :math:`\mathcal{T}_h` and the discrete space :math:`W_h` can be initialized by

.. code-block:: python

    a=AssemblerElement(m,ElementTriP1())

and the stiffness matrix and the load vector can be assembled by writing

.. code-block:: python

    A=a.iasm(lambda du,dv: du[0]*dv[0]+du[1]*dv[1])
    b=a.iasm(lambda v: 1.0*v)

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

