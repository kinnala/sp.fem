## \example examples_poisson.py
#
# In the following example we solve the Poisson equation in a three-dimensional
# box domain \f$\Omega=[0,1]^3\f$. We construct a simple manufactured solution
# and then compute the convergence rates
# of the finite element solution in the \f$H^1\f$-norm.
#
# We consider the following partial differential equation: find
# \f$u:\Omega\rightarrow\mathbb{R}\f$ that satisfies
# \f[
#     -\Delta u = f \quad \text{in $\Omega$,} \quad \text{and} \quad \frac{\partial u}{\partial n}=g-u \quad \text{on $\partial\Omega$.}
# \f]
# The weak formulation of the problem is: find \f$u\in H^1(\Omega)\f$ that satisfies
# \f[
#     (\nabla u, \nabla v) + \langle u, v \rangle = (f,v) + \langle g, v \rangle,
# \f]
# for every \f$v \in H^1(\Omega)\f$.
#
# Through symbolic computation of the Laplacian one can verify that
# the solution \f$u(x,y,z)=1+x-x^2y^2+x y z^3\f$
# is given by the following data:
# \f[
#     f(x,y,z)=2x^2+2y^2-6xyz, \quad \text{and}
#     \quad g(x,y,z)=\begin{cases}
#           3-3y^2+2yz^3 & \text{if $x=1$,} \\
#           -yz^3 & \text{if $x=0$,} \\
#           1+x-3x^2+2xz^3 & \text{if $y=1$,} \\
#           1+x-xz^3 & \text{if $y=0$,} \\
#           1+x+4xy-x^2y^2 & \text{if $z=1$,} \\
#           1+x-x^2y^2 & \text{if $z=0$.}
#     \end{cases}
# \f]
# In the following example, we compute the finite element solution \f$u_h\f$ using
# tetrahedral \f$P_k\f$ elements, \f$k \in \{1,2\}\f$. We evaluate the error
# \f$\|u-u_h\|_{H^1(\Omega)}\f$
# and verify the a priori error estimate
# \f[
#   \|u-u_h\|_{H^1(\Omega)} \leq C h^k |u|_2.
# \f]
# The code can be tested by starting `ipython --pylab` and running the following commands
#
#     import fem.examples_poisson
#     t=fem.examples_poisson.ExamplePoisson()
#     t.runTest(verbose=True)
