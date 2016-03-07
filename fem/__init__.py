## \mainpage Introduction
#
# Starting the refactoring, cleaning and adding of tests/examples.
#
# The major plan: test_module.py for full problem setup tests that
# simultaneously are used as documentation/examples.
#
# test_asm.py, test_mesh.py etc. for smaller and more concentrated unit tests.
#
# to-remove-list: AssemblerTri, GeometryComsol
#
# to-refactor-list: Elements (increase consistency in naming).
# GeometryPSLG2D->GeometryTriangle2D
#
# Steps:
# 1. Remove and change names.
# 2. Fix tests
# 3. Add few good examples/module level tests

## \page gettingstarted Getting started
#
# Add some simple example here.

## \page tips Tips
#
# \section Problems with Mayavi or other graphical toolbox
#
# Errors related to qt4, wx, mayavi, etc. can be sometimes
# fixed by simply changing environment variables or running
# ipython with the following flags
#
# ipython --gui=wx --pylab=wx
#
# ETS_TOOLKIT=qt4 ipython --gui=wx --pylab=wx
#
# \section Running the tests
#
# Simplest way is to discover all the tests using unittest as follows:
#
# ipython -m unittest discover ./fem
