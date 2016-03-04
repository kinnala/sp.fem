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
