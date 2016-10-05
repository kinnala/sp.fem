from distutils.core import setup
setup(
        name='spfem',
        packages=['spfem'],
        version='2016.1',
        install_requires=[
            "numpy",
            "scipy",
            "matplotlib",
            "sympy",
            "mayavi",
            ],
        description='Finite elements in pure SciPy',
        author='Tom Gustafsson',
        author_email='tom dot gustafsson at aalto dot fi',
        url='https://github.com/kinnala/sp.fem',
        download_url='https://github.com/kinnala/sp.fem/tarball/2016.1',
        keywords=['testing','logging','example'],
        classifiers=[],
        )
