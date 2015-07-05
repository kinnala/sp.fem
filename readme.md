# fem

In this repository you find the working draft of a flexible, fully-interpreted and yet well-performing finite element code written in scipy/Python.

## Description of files

### fem/geometry.py

Contains definitions of *Geometry* superclass and the respective subclasses. Geometry classes define the problem geometry explicitly (by mesh, CSG, ...) or implicitly (through some external file/program/procedure). Knowledge of the geometry is often required for, e.g., boundary conforming refines. Hence, Geometry classes work as *factories* that generate fixed meshes.

Simplest Geometries are GeometryMesh-classes that define the geometry through explicit meshes.

### fem/mesh.py

Meshes are outputted by Geometries and contain nodes, element connectivity information, element-to-face-mappings etc.

### fem/asm.py

Contains all finite element Assemblers that take (bi)linear forms and Meshes as inputs (and possibly additional info such as Elements) and output matrices.

## Running tests

Usually it suffices to write
```ipython -m unittest discover```
in the base directory.
