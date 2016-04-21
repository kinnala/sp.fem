# sp.fem

In this repository you find the working draft of a flexible, fully-interpreted and yet well-performing finite element code written in SciPy/Python.

The code is (c) Tom Gustafsson and licensed under AGPLv3. Windows binary of Triangle mesh generator is included for convenience. Triangle is (c) Jonathan Shewchuk and the license can be found in ./fem/triangle/LICENSE.

## Acknowledgements

The author and the code of sp.fem has strongly been influenced by the finite element knowledge shared by the following fellow scientists

* [A. Hannukainen](https://math.aalto.fi/en/current/publications/articles/?a%5b%5d=antti.hannukainen)
* [M. Juntunen](https://scholar.google.fi/citations?user=iKVJMwIAAAAJ)
* [A. Huhtala](http://arxiv.org/find/math/1/au:+Huhtala_A/0/1/0/all/0/1)

## Dependencies

* Python 2.7
* Numpy
* Scipy
* Sympy (Weakform module)
* Matplotlib
* Triangle (recommended; for 2D meshing)
* Mayavi (optional; for 3D mesh plots)

## Intallation: Linux

1. Install all dependencies (package manager of choice)
2. Compile Triangle binary to ./fem/triangle/triangle (https://www.cs.cmu.edu/~quake/triangle.html).
   An automatic installation and compilation script is provided for convenience under ./fem/triangle/.

## Installation: Windows 

1. Install Anaconda
2. triangle.exe is already found under ./fem/triangle/

## Automatic unit and performance tests

Tests are run automatically and can be found in [sp.fem-tests](https://github.com/kinnala/sp.fem-tests/) repository.

## Documentation

Currently doxygen must be run by the user. Use Doxyfile that is included in the repository.

## TODO list

* Implement ElementHdiv in 3D (2D exists)
* Implement ElementHcurl
* Investigate: Orientation of global facets/edges for better flexibility. Currently based on mesh index orders.
* Composition of elements for multiple scalar equations: vectorial, sum of elements.
* Adding more test cases, documentation, guards and speed tests.
* Adaptive 2D triangle meshing.
* Interpolation support in facet assembly.
* Assembly on subsets of elements
* Normal vectors in 3D facet assembly (2D works)
* Adding optimization flags to AssemblyElement (e.g. precompute global du,dv to save time in elements with large local stiffness matrix, or caching in Mapping and ElementH1 etc.)
* Tetgen (3D tetrahedral meshing) support and adaptive meshing.
* Loading of various mesh formats (GMSH, Comsol) in platform independent way.
* Plotting cross sections in MeshTet.
* Export to VTK for visualization in Paraview.
* Better interface ofor setting boundary conditions, current fetching of indices from Dofnum is horrible.
* Better way of defining complex weak forms (idea: class WeakForm which can be "compiled" to function handles)
* MeshLine and some 1D element. For example, Euler-Bernoulli beam element and ElementH2 (or ElementGlobal)
* Geometry for extrusion of 2D meshes to 3D meshes (mesh with TetGen maybe?)
* Geometry for rotation of 2D meshes/geometries to 3D meshes
* Change AssemblerElement to return other than assembled scipy.sparse-matrices (e.g. just (i,j,value) triples for combination of block matrices and later assembly, or for distribution to cluster using for example Petsc).
