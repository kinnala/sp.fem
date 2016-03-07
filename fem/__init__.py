## \page gettingstarted Getting started
#
# Add some simple example here.

## \page tips Tips
#
# \section tip1 Problems with Mayavi or other graphical toolbox
#
# Errors related to qt4, wx, mayavi, etc. can be sometimes
# fixed by simply changing environment variables or running
# ipython with the following flags
#
# ipython --gui=wx --pylab=wx
#
# ETS_TOOLKIT=qt4 ipython --gui=wx --pylab=wx
#
# \section tip2 Running the tests
#
# Simplest way is to discover all the tests using unittest as follows:
#
# ipython -m unittest discover ./fem
