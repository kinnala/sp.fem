# sp.fem

In this repository you find the working draft of a flexible, fully-interpreted and yet well-performing finite element code written in SciPy/Python.

The code is initially (c) Tom Gustafsson but the license will change in the future. Triangle is (c) Jonathan Shewchuk.

## Dependencies

* Python 2.7
* Numpy
* Scipy
* Matplotlib
* Triangle (recommended; for 2D meshing)
* Mayavi (optional; for 3D mesh plots)

## Intallation: Linux

1. Install all dependencies (package manager of choice)
2. Compile Triangle binary to ./fem/triangle/triangle (https://www.cs.cmu.edu/~quake/triangle.html)

## Installation: Windows 

1. Install Anaconda
3. triangle.exe is already found under ./fem/triangle/

## TODO list

### Possibly breaking changes (solve first)

* Implement ElementHdiv (mostly to check consistency of Element and Mapping interfaces for arbitrary elements)
* Implement ElementHcurl
* Orientation of global facets/edges for better flexibility.
* Composition of elements for multiple scalar equations: vectorial, sum of elements

### First things after fixing breaking changes

* Adding more test cases, documentation, guards + speed tests.
* Adaptive 2D triangle meshing.

### Other ideas

* Assemblies on subsets of elements
* Interpolation in assembly
* Normal vectors in facet assemblies
* Adding optimization flags to AssemblyElement (e.g. disable computation of ddu)
* Adding more test cases, documentation, guards + speed tests.
* Tetgen support and adaptive meshing.
* Loading of various mesh formats (GMSH, Comsol) in platform independent way
* Plotting cross sections in MeshTet.
* Better way of setting boundary conditions, current fetching of indices from Dofnum is horrible
* Better way of defining complex weak forms (idea: class WeakForm which is "compiled" to a simple function handle for use)
* MeshLine and some 1D element. For example, Euler-Bernoulli beam element and ElementH2 (or ElementGlobal)
* Geometry for extrusion of 2D meshes to 3D meshes (mesh with TetGen maybe?)
* Geometry for rotation of 2D meshes/geometries to 3D meshes

## Running tests

Usually it suffices to write
```ipython2 --gui=wx --pylab=wx -m unittest discover ./fem```
or
```python -m unittest discover -v ./fem```
in the base directory. Note that newer Mayavi versions might crash if ```--gui=wx --pylab==wx``` is omitted.

## iPython usage in Linux
Run iPython using
```ipython --gui=wx --pylab=wx```

## iPython usage in Windows
Mayavi does not work under windows without
```export QT_API=pyqt```
and iPython lags without
```ipython --pylab```

## Poisson assemble speed in 2D (22.2.16)
Done using AssemblerElement and ElementP1. Computer is Core i7 950 at 3.07 GHz and 4 GB of RAM. Windows 7 with Anaconda Scipy distribution.

|NDOF |time (s)|
|-----|------|
|1089 |0.0113|
|4225 |0.0390|
|16641|0.1820|
|66049|0.8950|
|263169|3.73|
|1050625|14.4|

