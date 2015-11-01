# sp.fem

In this repository you find the working draft of a flexible, fully-interpreted and yet well-performing finite element code written in SciPy/Python.

The code is initially (c) Tom Gustafsson but the license will change in the future.

# Dependencies

* Python 2.7
* Numpy
* Scipy
* Matplotlib
* Shapely (optional; for 2D geometries)
* Triangle (optional; for 2D meshing)
* Mayavi (optional; for gl plots)

# Linux

1. Install all dependencies (package manager of choice)
2. Compile Triangle binary to ./fem/triangle/triangle (https://www.cs.cmu.edu/~quake/triangle.html)

# Windows 

1. Install Anaconda
2. Install Shapely as follows:
* Get whl-file from http://www.lfd.uci.edu/~gohlke/pythonlibs/
* Open Anaconda Console: *conda run cmd*
* Go to the directory with the downloaded whl-file and run *pip install [filename]*
3. triangle.exe is already found under ./fem/triangle/

## TODO list

* Interpolation in AssemblerTriP1.fasm
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
