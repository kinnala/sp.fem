# sp.fem

In this repository you find the working draft of a flexible, fully-interpreted and yet well-performing finite element code written in SciPy/Python.

The code is initially (c) Tom Gustafsson but the license will change in the future. Triangle is (c) Jonathan Shewchuk.

## Dependencies

* Python 2.7
* Numpy
* Scipy
* Matplotlib
* Triangle (recommended; for 2D meshing)
* Mayavi (optional; for gl plots)

## Intallation: Linux

1. Install all dependencies (package manager of choice)
2. Compile Triangle binary to ./fem/triangle/triangle (https://www.cs.cmu.edu/~quake/triangle.html)

## Installation: Windows 

1. Install Anaconda
3. triangle.exe is already found under ./fem/triangle/

## TODO list

* Interpolation in AssemblerTriP1.fasm
* AssemblerTetP1 and MeshTet and GeometryMeshTet
* AssemblerLineP1, MeshLine?
* AssemblerElement (and Element classes)

## Running tests

Usually it suffices to write
```ipython2 -m unittest discover ./fem```
or
```python -m unittest discover -v ./fem```
in the base directory.

## iPython usage in Windows
Mayavi does not work under windows without
```export QT_API=pyqt```
and iPython lags without
```ipython --pylab```
