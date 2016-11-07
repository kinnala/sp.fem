"""
For backwards compatiblity.
"""
import sys
import spfem.assembly

sys.modules[__name__] = sys.modules['spfem.assembly']
