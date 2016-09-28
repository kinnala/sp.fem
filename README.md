# sp.fem

[![Build Status](https://travis-ci.org/kinnala/sp.fem.svg)](https://travis-ci.org/kinnala/sp.fem) [![codecov](https://codecov.io/gh/kinnala/sp.fem/branch/master/graph/badge.svg)](https://codecov.io/gh/kinnala/sp.fem)

In this repository you find the working draft of a flexible, fully-interpreted and yet well-performing finite element code written in SciPy/Python.

The code is (c) Tom Gustafsson and licensed under AGPLv3. Triangle is (c) Jonathan Shewchuk and the license can be found in ./fem/triangle/LICENSE.

## Acknowledgements

The author and the code of sp.fem has strongly been influenced by the finite element knowledge shared by the following fellow scientists

* [A. Hannukainen](https://math.aalto.fi/en/current/publications/articles/?a%5b%5d=antti.hannukainen)
* [M. Juntunen](https://scholar.google.fi/citations?user=iKVJMwIAAAAJ)
* [A. Huhtala](http://arxiv.org/find/math/1/au:+Huhtala_A/0/1/0/all/0/1)

Some parts of the vectorized code are more or less directly influenced by the efficient vectorized Matlab implementations written by these people. This library would either not exists or at least be ten times slower if not for them.

## Documentation

The documentation is built using Sphinx and can currently be found under /docs.
