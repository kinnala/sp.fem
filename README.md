# (DEPRECATED) sp.fem

[![Build Status](https://travis-ci.org/kinnala/sp.fem.svg)](https://travis-ci.org/kinnala/sp.fem) [![codecov](https://codecov.io/gh/kinnala/sp.fem/branch/master/graph/badge.svg)](https://codecov.io/gh/kinnala/sp.fem)

The code has been renamed, relicensed and ported to Python 3. Find the latests developments [here](https://github.com/kinnala/scikit-fem). This Python 2 version is not updated anymore.

In this repository you find the working draft of a lightweight,
fully-interpreted and yet well-performing finite element code written in
SciPy/Python.

The code is (c) Tom Gustafsson and licensed under AGPLv3.

## Ideology

The main task of the library is to perform *finite element assembly* for various
types of *finite elements* and *meshes*. Thus, the core functionality can be
summarized in the following heuristic identity:

```
Mesh (spfem) + Bilinear form (function) + Element (spfem) = Sparse matrix (scipy)
```

This implies that the library is not suitable for developers who wish to have a
fully blackbox PDE solver. A basic understanding of finite elements is required
in order to successfully apply the resulting matrices.

## Minimal example
The following code solves the Poisson equation in a unit square with zero
boundary conditions and unit loading.
```python
from spfem.mesh import MeshTri
from spfem.assembly import AssemblerElement
from spfem.element import ElementTriP1
from spfem.utils import direct

m = MeshTri()
m.refine(6)

a = AssemblerElement(m, ElementTriP1())

A = a.iasm(lambda du, dv: du[0]*dv[0] + du[1]*dv[1])
b = a.iasm(lambda v: 1.0*v)

x = direct(A, b, I=m.interior_nodes())

m.plot3(x)
m.show()
```

## Acknowledgements

The author and the code of sp.fem has strongly been influenced by the finite
element knowledge shared by the following fellow scientists

* [A. Hannukainen](https://math.aalto.fi/en/current/publications/articles/?a%5b%5d=antti.hannukainen)
* [M. Juntunen](https://scholar.google.fi/citations?user=iKVJMwIAAAAJ)
* [A. Huhtala](http://arxiv.org/find/math/1/au:+Huhtala_A/0/1/0/all/0/1)

Some parts of the vectorized code are more or less directly influenced by the
efficient vectorized Matlab implementations written by these people. This
library would either not exists or at least be ten times slower if not for
them.

## Documentation

The documentation is built using Sphinx and can be found at [Github
Pages](http://kinnala.github.io/sp.fem/).
