# sp.fem

In this repository you find the working draft of a flexible, fully-interpreted and yet well-performing finite element code written in SciPy/Python.

The code is initially (c) Tom Gustafsson but the license will change in the future.

## TODO list

* Interpolation in AssemblerTriP1
* AssemblerTetP1 and MeshTet and GeometryMeshTet
* AssemblerLineP1, MeshLine?
* AssemblerElement (and Element classes)

## Running tests

Usually it suffices to write
```ipython -m unittest discover ./fem```
in the base directory.

Also, e.g,
```ipython -m unittest fem.test_mesh```
is fine for running a single test module.
