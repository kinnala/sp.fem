# sp.fem

In this repository you find the working draft of a flexible, fully-interpreted and yet well-performing finite element code written in SciPy/Python.

The code is initially (c) Tom Gustafsson but the license will change in the future.

## Description of files

### fem/geometry.py

Contains definitions of *Geometry* superclass and the respective subclasses. Geometry classes define the problem geometry explicitly (by mesh, CSG, ...) or implicitly (through some external file/program/procedure). Knowledge of the geometry is often required for, e.g., boundary conforming refines. Hence, Geometry classes work as *factories* that generate fixed meshes.

Simplest Geometries are GeometryMesh-classes that define the geometry through explicit meshes.

### fem/mesh.py

Meshes are outputted by Geometries and contain nodes, element connectivity information, element-to-face-mappings etc.

### fem/asm.py

Contains all finite element Assemblers that take (bi)linear forms and Meshes as inputs (and possibly additional info such as Elements) and output matrices.

### fem/mapping.py

Mappings such as affine mappings, isoparametric mappings, ... ?

### fem/element.py

TODO

### fem/quadrature.py

TODO

## TODO list

* facet assembly in AssemblerTriP1
* global coordinates(ok) and interpolation in AssemblerTriP1
* AssemblerTetP1 and MeshTet and GeometryMeshTet
* AssemblerElement (and Element classes)

## Running tests

Usually it suffices to write
```ipython -m unittest discover```
in the base directory.

Also, e.g,
```ipython -m unittest fem.test_mesh```
is fine for running a single test module.
