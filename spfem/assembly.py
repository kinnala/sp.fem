# -*- coding: utf-8 -*-
"""
Assembly of matrices related to linear and bilinear forms.

Examples
--------
Assemble the stiffness matrix related to
the Poisson problem using the piecewise linear elements.

.. code-block:: python

    from spfem.mesh import MeshTri
    from spfem.assembly import AssemblerElement
    from spfem.element import ElementTriP1

    m = MeshTri()
    m.refine(3)
    e = ElementTriP1()
    a = AssemblerElement(m, e)

    def bilinear_form(du, dv):
        return du[0]*dv[0] + du[1]*dv[1]

    K = a.iasm(bilinear_form)
"""
import numpy as np
import inspect
import abc
from scipy.sparse import coo_matrix

import spfem.mesh
import spfem.mapping
from spfem.quadrature import get_quadrature
from spfem.utils import const_cell, cell_shape

class Assembler(object):
    """Finite element assembler."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    def fillargs(self, oldform, newargs):
        """Used for filling functions with required set of arguments."""
        oldargs = inspect.getargspec(oldform).args
        if oldargs == newargs:
            # the given form already has correct arguments
            return oldform

        y = []
        for oarg in oldargs:
            # add corresponding new argument index to y for
            # each old argument
            for ix, narg in enumerate(newargs):
                if oarg == narg:
                    y.append(ix)
                    break

        if len(oldargs) == 1:
            def newform(*x):
                return oldform(x[y[0]])
        elif len(oldargs) == 2:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]])
        elif len(oldargs) == 3:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]])
        elif len(oldargs) == 4:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]])
        elif len(oldargs) == 5:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]])
        elif len(oldargs) == 6:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]])
        elif len(oldargs) == 7:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]])
        elif len(oldargs) == 8:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]])
        elif len(oldargs) == 9:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]])
        elif len(oldargs) == 10:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]])
        elif len(oldargs) == 11:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]],
                               x[y[10]])
        elif len(oldargs) == 12:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]],
                               x[y[10]], x[y[11]])
        elif len(oldargs) == 13:
            def newform(*x):
                return oldform(x[y[0]], x[y[1]], x[y[2]], x[y[3]], x[y[4]],
                               x[y[5]], x[y[6]], x[y[7]], x[y[8]], x[y[9]],
                               x[y[10]], x[y[11]], x[y[12]])
        else:
            raise NotImplementedError("Maximum number of arguments reached.")

        return newform


class AssemblerAbstract(Assembler):
    """An assembler for elements of type
    :class:`spfem.element.AbstractElement`.

    These elements are defined through degrees of freedom but
    are not limited to H^1-conforming elements. As a result,
    this assembler is more computationally intensive than
    :class:`spfem.assembly.AssemblerElement` so use it instead
    if possible.

    Parameters
    ----------
    mesh : :class:`spfem.mesh.Mesh`
        The finite element mesh.

    elem_u : :class:`spfem.element.Element`
        The element for the solution function.

    elem_v : (OPTIONAL) :class:`spfem.element.Element`
        The element for the test function. By default, same element
        is used for both.

    intorder : (OPTIONAL) int
        The used quadrature order.

        The basis functions at quadrature points are precomputed
        in initializer. By default, the order of quadrature rule
        is deduced from the maximum polynomial degree of an element.
    """
    def __init__(self, mesh, elem_u, elem_v=None, intorder=None):
        if not isinstance(mesh, spfem.mesh.Mesh):
            raise Exception("First parameter must be an instance of "
                            "spfem.mesh.Mesh.")
        if not isinstance(elem_u, spfem.element.AbstractElement):
            raise Exception("Second parameter must be an instance of "
                            "spfem.element.AbstractElement.")
        if elem_v is not None:
            if not isinstance(elem_v, spfem.element.AbstractElement):
                raise Exception("Third parameter must be an instance of "
                                "spfem.element.AbstractElement.")

        self.mesh = mesh
        self.elem_u = elem_u
        self.dofnum_u = Dofnum(mesh, elem_u)
        self.mapping = mesh.mapping()

        # duplicate test function element type if None is given
        if elem_v is None:
            self.elem_v = elem_u
            self.dofnum_v = self.dofnum_u
        else:
            self.elem_v = elem_v
            self.dofnum_v = Dofnum(mesh, elem_v)

        if intorder is None:
            # compute the maximum polynomial degree from elements
            self.intorder = self.elem_u.maxdeg + self.elem_v.maxdeg
        else:
            self.intorder = intorder

        # quadrature points and weights
        X, _ = get_quadrature(self.mesh.refdom, self.intorder)
        # global quadrature points
        x = self.mapping.F(X, range(self.mesh.t.shape[1]))
        # pre-compute basis functions at quadrature points
        self.u, self.du, self.ddu = self.elem_u.evalbasis(self.mesh, x)

        if elem_v is None:
            self.v = self.u
            self.dv = self.du
            self.ddv = self.ddu
        else:
            self.v, self.dv, self.ddv = self.elem_v.evalbasis(self.mesh, x)

    def iasm(self, form, tind=None, interp=None):
        if tind is None:
            # assemble on all elements by default
            tind = range(self.mesh.t.shape[1])
        nt = len(tind)

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams or 'ddu' in oldparams:
            paramlist = ['u', 'v', 'du', 'dv', 'ddu', 'ddv', 'x', 'w', 'h']
            bilinear = True
        else:
            paramlist = ['v', 'dv', 'ddv', 'x', 'w', 'h']
            bilinear = False
        fform = self.fillargs(form, paramlist)

        # quadrature points and weights
        X, W = get_quadrature(self.mesh.refdom, self.intorder)

        # global quadrature points
        x = self.mapping.F(X, tind)

        # jacobian at quadrature points
        detDF = self.mapping.detDF(X, tind)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        Nbfun_v = self.dofnum_v.t_dof.shape[0]

        # interpolate some previous discrete function at quadrature points
        w = {}
        if interp is not None:
            if not isinstance(interp, dict):
                raise Exception("The input solution vector(s) must be in a "
                                "dictionary! Pass e.g. {0:u} instead of u.")
            # interpolate the solution vectors at quadrature points
            zero = 0.0*x[0]
            w = {}
            for k in interp:
                w[k] = zero
                for j in range(self.Nbfun_u):
                    jdofs = self.dofnum_u.t_dof[j, :]
                    w[k] += interp[k][jdofs][:, None]*self.u[j]

        # compute the mesh parameter from jacobian determinant
        h = np.abs(detDF)**(1.0/self.mesh.dim())

        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_u*Nbfun_v*nt)
            rows = np.zeros(Nbfun_u*Nbfun_v*nt)
            cols = np.zeros(Nbfun_u*Nbfun_v*nt)

            for j in range(Nbfun_u):
                for i in range(Nbfun_v):
                    # find correct location in data,rows,cols
                    ixs = slice(nt*(Nbfun_v*j + i), nt*(Nbfun_v*j + i + 1))

                    # compute entries of local stiffness matrices
                    data[ixs] = np.dot(fform(self.u[j], self.v[i],
                                             self.du[j], self.dv[i],
                                             self.ddu[j], self.ddv[i],
                                             x, w, h)*np.abs(detDF), W)
                    rows[ixs] = self.dofnum_v.t_dof[i, tind]
                    cols[ixs] = self.dofnum_u.t_dof[j, tind]

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, self.dofnum_u.N)).tocsr()

        else:
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_v*nt)
            rows = np.zeros(Nbfun_v*nt)
            cols = np.zeros(Nbfun_v*nt)

            for i in range(Nbfun_v):
                # find correct location in data,rows,cols
                ixs = slice(nt*i, nt*(i + 1))

                # compute entries of local stiffness matrices
                data[ixs] = np.dot(fform(self.v[i], self.dv[i], self.ddv[i],
                                         x, w, h)*np.abs(detDF), W)
                rows[ixs] = self.dofnum_v.t_dof[i, :]
                cols[ixs] = np.zeros(nt)

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, 1)).toarray().T[0]

    def inorm(self, form, interp, intorder=None):
        # evaluate norm on all elements
        tind = range(self.mesh.t.shape[1])

        if not isinstance(interp, dict):
            raise Exception("The input solution vector(s) must be in a "
                            "dictionary! Pass e.g. {0:u} instead of u.")

        if intorder is None:
            intorder = 2*self.elem_u.maxdeg

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        paramlist = ['u', 'du', 'ddu', 'x', 'h']
        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh.refdom, intorder)

        # mappings
        x = self.mapping.F(X, tind) # reference facet to global facet

        # jacobian at quadrature points
        detDF = self.mapping.detDF(X, tind)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        dim = self.mesh.p.shape[0]

        # interpolate the solution vectors at quadrature points
        zero = 0.0*x[0]
        w, dw, ddw = ({} for i in range(3))
        for k in interp:
            w[k] = zero
            dw[k] = const_cell(zero, dim)
            ddw[k] = const_cell(zero, dim, dim)
            for j in range(Nbfun_u):
                jdofs = self.dofnum_u.t_dof[j, :]
                w[k] += interp[k][jdofs][:, None]\
                        * self.u[j]
                for a in range(dim):
                    dw[k][a] += interp[k][jdofs][:, None]\
                                * self.du[j][a]
                    for b in range(dim):
                        ddw[k][a][b] += interp[k][jdofs][:, None]\
                                        * self.ddu[j][a][b]

        # compute the mesh parameter from jacobian determinant
        h = np.abs(detDF)**(1.0/self.mesh.dim())

        return np.dot(fform(w, dw, ddw, x, h)**2*np.abs(detDF), W)

    def fnorm(self, form, interp, intorder=None, interior=False, normals=True):
        if interior:
            # evaluate norm on all interior facets
            find = self.mesh.interior_facets()
        else:
            # evaluate norm on all boundary facets
            find = self.mesh.boundary_facets()

        if not isinstance(interp, dict):
            raise Exception("The input solution vector(s) must be in a "
                            "dictionary! Pass e.g. {0:u} instead of u.")

        if intorder is None:
            intorder = 2*self.elem_u.maxdeg

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if 'u1' in oldparams or 'du1' in oldparams or 'ddu1' in oldparams:
            if interior is False:
                raise Exception("The supplied form contains u1 although "
                                "no interior=True is given.")
        if interior:
            paramlist = ['u1', 'u2', 'du1', 'du2', 'ddu1', 'ddu2',
                         'x', 'n', 't', 'h']
        else:
            paramlist = ['u', 'du', 'ddu', 'x', 'n', 't', 'h']
        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh.brefdom, intorder)

        # indices of elements at different sides of facets
        tind1 = self.mesh.f2t[0, find]
        tind2 = self.mesh.f2t[1, find]

        # mappings
        x = self.mapping.G(X, find=find) # reference facet to global facet
        detDG = self.mapping.detDG(X, find)

        # compute basis function values at quadrature points
        u1, du1, ddu1 = self.elem_u.evalbasis(self.mesh, x, tind=tind1)
        if interior:
            u2, du2, ddu2 = self.elem_u.evalbasis(self.mesh, x, tind=tind2)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        dim = self.mesh.p.shape[0]

        n = {}
        t = {}
        if normals:
            Y = self.mapping.invF(x, tind=tind1) # global facet to ref element
            n = self.mapping.normals(Y, tind1, find, self.mesh.t2f)
            if len(n) == 2: # TODO fix for 3D and other than triangles?
                t[0] = -n[1]
                t[1] = n[0]

        # interpolate the solution vectors at quadrature points
        zero = np.zeros((len(find), len(W)))
        w1, dw1, ddw1 = ({} for i in range(3))
        if interior:
            w2, dw2, ddw2 = ({} for i in range(3))
        for k in interp:
            w1[k] = zero
            dw1[k] = const_cell(zero, dim)
            ddw1[k] = const_cell(zero, dim, dim)
            if interior:
                w2[k] = zero
                dw2[k] = const_cell(zero, dim)
                ddw2[k] = const_cell(zero, dim, dim)
            for j in range(Nbfun_u):
                jdofs1 = self.dofnum_u.t_dof[j, tind1]
                jdofs2 = self.dofnum_u.t_dof[j, tind2]
                w1[k] += interp[k][jdofs1][:, None] * u1[j]
                if interior:
                    w2[k] += interp[k][jdofs2][:, None] * u2[j]
                for a in range(dim):
                    dw1[k][a] += interp[k][jdofs1][:, None]\
                                 * du1[j][a]
                    if interior:
                        dw2[k][a] += interp[k][jdofs2][:, None]\
                                     * du2[j][a]
                    for b in range(dim):
                        ddw1[k][a][b] += interp[k][jdofs1][:, None]\
                                         * ddu1[j][a][b]
                        if interior:
                            ddw2[k][a][b] += interp[k][jdofs2][:, None]\
                                             * ddu2[j][a][b]

        h = np.abs(detDG)**(1.0/(self.mesh.dim()-1.0))

        if interior:
            return np.dot(fform(w1, w2, dw1, dw2, ddw1, ddw2,
                                x, n, t, h)**2*np.abs(detDG), W), find
        else:
            return np.dot(fform(w1, dw1, ddw1,
                                x, n, t, h)**2*np.abs(detDG), W), find


class AssemblerElement(Assembler):
    """An assembler for Element classes.

    These elements are defined through reference elements
    and are limited to H^1-conforming elements.

    Parameters
    ----------
    mesh : :class:`spfem.mesh.Mesh`
        The finite element mesh.

    elem_u : :class:`spfem.element.Element`
        The element for the solution function.

    elem_v : (OPTIONAL) :class:`spfem.element.Element`
        The element for the test function. By default,
        the same element is used for both.

    mapping : (OPTIONAL) :class:`spfem.mapping.Mapping`
        The mesh will give some sort of default mapping but sometimes, e.g.
        when using isoparametric elements, the user might have to provide
        a different mapping.
    """
    def __init__(self, mesh, elem_u, elem_v=None, mapping=None):
        if not isinstance(mesh, spfem.mesh.Mesh):
            raise Exception("First parameter must be an instance of "
                            "spfem.mesh.Mesh!")
        if not isinstance(elem_u, spfem.element.Element):
            raise Exception("Second parameter must be an instance of "
                            "spfem.element.Element!")

        # get default mapping from the mesh
        if mapping is None:
            self.mapping = mesh.mapping()
        else:
            self.mapping = mapping # assumes an already initialized mapping

        self.mesh = mesh
        self.elem_u = elem_u
        self.dofnum_u = Dofnum(mesh, elem_u)

        # duplicate test function element type if None is given
        if elem_v is None:
            self.elem_v = elem_u
            self.dofnum_v = self.dofnum_u
        else:
            self.elem_v = elem_v
            self.dofnum_v = Dofnum(mesh, elem_v)

    def iasm(self, form, intorder=None, tind=None, interp=None):
        """Return a matrix related to a bilinear or linear form
        where the integral is over the interior of the domain.

        Parameters
        ----------
        form : function handle
            The bilinear or linear form function handle.
            The supported parameters can be found in the
            following table.

            +-----------+----------------------+--------------+
            | Parameter | Explanation          | Supported in |
            +-----------+----------------------+--------------+
            | u         | solution             | bilinear     |
            +-----------+----------------------+--------------+
            | v         | test fun             | both         |
            +-----------+----------------------+--------------+
            | du        | solution derivatives | bilinear     |
            +-----------+----------------------+--------------+
            | dv        | test fun derivatives | both         |
            +-----------+----------------------+--------------+
            | x         | spatial location     | both         |
            +-----------+----------------------+--------------+
            | w         | cf. interp           | both         |
            +-----------+----------------------+--------------+
            | h         | the mesh parameter   | both         |
            +-----------+----------------------+--------------+

            The function handle must use these exact names for
            the variables. Unused variable names can be omitted.

            Examples of valid bilinear forms:
            ::

                def bilin_form1(du, dv):
                    # Note that the element must be
                    # defined for two-dimensional
                    # meshes for this to make sense!
                    return du[0]*dv[0] + du[1]*dv[1]

                def bilin_form2(du, v):
                    return du[0]*v

                bilin_form3 = lambda u, v, x: x[0]**2*u*v

            Examples of valid linear forms:
            ::

                def lin_form1(v):
                    return v

                def lin_form2(h, x, v):
                    import numpy as np
                    mesh_parameter = h
                    X = x[0]
                    Y = x[1]
                    return mesh_parameter*np.sin(np.pi*X)*np.sin(np.pi*Y)*v

            The linear forms are automatically detected to be
            non-bilinear through the omission of u or du.

        intorder : int
            The order of polynomials for which the applied
            quadrature rule is exact. By default,
            2*Element.maxdeg is used. Reducing this number
            can sometimes reduce the computation time.

        interp : numpy array
            Using this flag, the user can provide
            a solution vector that is interpolated
            to the quadrature points and included in
            the computation of the bilinear form
            (the variable w). Useful e.g. when solving
            nonlinear problems.
        """
        if tind is None:
            # assemble on all elements by default
            tind = range(self.mesh.t.shape[1])
        nt = len(tind)
        if intorder is None:
            # compute the maximum polynomial degree from elements
            intorder = self.elem_u.maxdeg + self.elem_v.maxdeg

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if 'u' in oldparams or 'du' in oldparams:
            paramlist = ['u', 'v', 'du', 'dv', 'x', 'w', 'h']
            bilinear = True
        else:
            paramlist = ['v', 'dv', 'x', 'w', 'h']
            bilinear = False
        fform = self.fillargs(form, paramlist)

        # quadrature points and weights
        X, W = get_quadrature(self.mesh.refdom, intorder)

        # global quadrature points
        x = self.mapping.F(X, tind)

        # jacobian at quadrature points
        detDF = self.mapping.detDF(X, tind)

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        Nbfun_v = self.dofnum_v.t_dof.shape[0]

        # interpolate some previous discrete function at quadrature points
        w = {}
        if interp is not None:
            if not isinstance(interp, dict):
                raise Exception("The input solution vector(s) must be in a "
                                "dictionary! Pass e.g. {0:u} instead of u.")
            for k in interp:
                w[k] = 0.0*x[0]
                for j in range(Nbfun_u):
                    phi, _ = self.elem_u.lbasis(X, j)
                    w[k] += np.outer(interp[k][self.dofnum_u.t_dof[j, :]], phi)

        # compute the mesh parameter from jacobian determinant
        h = np.abs(detDF)**(1.0/self.mesh.dim())

        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_u*Nbfun_v*nt)
            rows = np.zeros(Nbfun_u*Nbfun_v*nt)
            cols = np.zeros(Nbfun_u*Nbfun_v*nt)

            for j in range(Nbfun_u):
                u, du = self.elem_u.gbasis(self.mapping, X, j, tind)
                for i in range(Nbfun_v):
                    v, dv = self.elem_v.gbasis(self.mapping, X, i, tind)

                    # find correct location in data,rows,cols
                    ixs = slice(nt*(Nbfun_v*j+i), nt*(Nbfun_v*j+i+1))

                    # compute entries of local stiffness matrices
                    data[ixs] = np.dot(fform(u, v, du, dv, x, w, h)
                                       * np.abs(detDF), W)
                    rows[ixs] = self.dofnum_v.t_dof[i, tind]
                    cols[ixs] = self.dofnum_u.t_dof[j, tind]

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, self.dofnum_u.N)).tocsr()

        else:
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_v*nt)
            rows = np.zeros(Nbfun_v*nt)
            cols = np.zeros(Nbfun_v*nt)

            for i in range(Nbfun_v):
                v, dv = self.elem_v.gbasis(self.mapping, X, i, tind)

                # find correct location in data,rows,cols
                ixs = slice(nt*i, nt*(i+1))

                # compute entries of local stiffness matrices
                data[ixs] = np.dot(fform(v, dv, x, w, h)*np.abs(detDF), W)
                rows[ixs] = self.dofnum_v.t_dof[i, :]
                cols[ixs] = np.zeros(nt)

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, 1)).toarray().T[0]

    def fasm(self, form, find=None, interior=False, intorder=None,
             normals=True, interp=None):
        """Facet assembly."""
        if find is None:
            if interior:
                find = self.mesh.interior_facets()
            else:
                find = self.mesh.boundary_facets()

        if intorder is None:
            intorder = self.elem_u.maxdeg + self.elem_v.maxdeg

        nv = self.mesh.p.shape[1]
        nt = self.mesh.t.shape[1]
        ne = find.shape[0]

        # check and fix parameters of form
        oldparams = inspect.getargspec(form).args
        if interior:
            if 'u1' in oldparams or 'du1' in oldparams:
                paramlist = ['u1', 'u2', 'v1', 'v2',
                             'du1', 'du2', 'dv1', 'dv2',
                             'x', 'h', 'n', 'w']
                bilinear = True
            else:
                paramlist = ['v1', 'v2', 'dv1', 'dv2',
                             'x', 'h', 'n', 'w']
                bilinear = False
        else:
            if 'u' in oldparams or 'du' in oldparams:
                paramlist = ['u', 'v', 'du', 'dv', 'x', 'h', 'n', 'w']
                bilinear = True
            else:
                paramlist = ['v', 'dv', 'x', 'h', 'n', 'w']
                bilinear = False
        fform = self.fillargs(form, paramlist)

        X, W = get_quadrature(self.mesh.brefdom, intorder)

        # boundary element indices
        tind1 = self.mesh.f2t[0, find]
        tind2 = self.mesh.f2t[1, find]

        # mappings
        x = self.mapping.G(X, find=find) # reference facet to global facet
        Y1 = self.mapping.invF(x, tind=tind1) # global facet to ref element
        Y2 = self.mapping.invF(x, tind=tind2) # global facet to ref element

        Nbfun_u = self.dofnum_u.t_dof.shape[0]
        Nbfun_v = self.dofnum_v.t_dof.shape[0]

        detDG = self.mapping.detDG(X, find)

        # compute normal vectors
        n = {}
        if normals:
            # normals based on tind1 only
            n = self.mapping.normals(Y1, tind1, find, self.mesh.t2f)

        # compute the mesh parameter from jacobian determinant
        if self.mesh.dim() > 1.0:
            h = np.abs(detDG)**(1.0/(self.mesh.dim() - 1.0))
        else: # exception for 1D mesh (no boundary h defined)
            h = None

        # interpolate some previous discrete function at quadrature points
        w = {}
        if interp is not None:
            if not isinstance(interp, dict):
                raise Exception("The input solution vector(s) must be in a "
                                "dictionary! Pass e.g. {0:u} instead of u.")
            for k in interp:
                w[k] = 0.0*x[0]
                for j in range(Nbfun_u):
                    phi, _ = self.elem_u.gbasis(self.mapping, Y1, j, tind1)
                    w[k] += interp[k][self.dofnum_u.t_dof[j, tind1], None]*phi

        # bilinear form
        if bilinear:
            # initialize sparse matrix structures
            ndata = Nbfun_u*Nbfun_v*ne
            if interior:
                data = np.zeros(4*ndata)
                rows = np.zeros(4*ndata)
                cols = np.zeros(4*ndata)
            else:
                data = np.zeros(ndata)
                rows = np.zeros(ndata)
                cols = np.zeros(ndata)

            for j in range(Nbfun_u):
                u1, du1 = self.elem_u.gbasis(self.mapping, Y1, j, tind1)
                if interior:
                    u2, du2 = self.elem_u.gbasis(self.mapping, Y2, j, tind2)
                    if j == 0:
                        # these are zeros corresponding to the shapes of u,du
                        z = const_cell(0, *cell_shape(u2))
                        dz = const_cell(0, *cell_shape(du2))
                for i in range(Nbfun_v):
                    v1, dv1 = self.elem_v.gbasis(self.mapping, Y1, i, tind1)
                    if interior:
                        v2, dv2 = self.elem_v.gbasis(self.mapping, Y2, i, tind2)

                    # compute entries of local stiffness matrices
                    if interior:
                        ixs1 = slice(ne*(Nbfun_v*j + i),
                                     ne*(Nbfun_v*j + i + 1))
                        ixs2 = slice(ne*(Nbfun_v*j + i) + ndata,
                                     ne*(Nbfun_v*j + i + 1) + ndata)
                        ixs3 = slice(ne*(Nbfun_v*j + i) + 2*ndata,
                                     ne*(Nbfun_v*j + i + 1) + 2*ndata)
                        ixs4 = slice(ne*(Nbfun_v*j + i) + 3*ndata,
                                     ne*(Nbfun_v*j + i + 1) + 3*ndata)

                        data[ixs1] = np.dot(fform(u1, z, v1, z,
                                                  du1, dz, dv1, dz,
                                                  x, h, n, w)*np.abs(detDG), W)
                        rows[ixs1] = self.dofnum_v.t_dof[i, tind1]
                        cols[ixs1] = self.dofnum_u.t_dof[j, tind1]

                        data[ixs2] = np.dot(fform(z, u2, z, v2,
                                                  dz, du2, dz, dv2,
                                                  x, h, n, w)*np.abs(detDG), W)
                        rows[ixs2] = self.dofnum_v.t_dof[i, tind2]
                        cols[ixs2] = self.dofnum_u.t_dof[j, tind2]

                        data[ixs3] = np.dot(fform(z, u2, v1, z,
                                                  dz, du2, dv1, dz,
                                                  x, h, n, w)*np.abs(detDG), W)
                        rows[ixs3] = self.dofnum_v.t_dof[i, tind1]
                        cols[ixs3] = self.dofnum_u.t_dof[j, tind2]

                        data[ixs4] = np.dot(fform(u1, z, z, v2,
                                                  du1, dz, dz, dv2,
                                                  x, h, n, w)*np.abs(detDG), W)
                        rows[ixs4] = self.dofnum_v.t_dof[i, tind2]
                        cols[ixs4] = self.dofnum_u.t_dof[j, tind1]
                    else:
                        ixs = slice(ne*(Nbfun_v*j + i), ne*(Nbfun_v*j + i + 1))
                        data[ixs] = np.dot(fform(u1, v1, du1, dv1,
                                                 x, h, n, w)*np.abs(detDG), W)
                        rows[ixs] = self.dofnum_v.t_dof[i, tind1]
                        cols[ixs] = self.dofnum_u.t_dof[j, tind1]

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, self.dofnum_u.N)).tocsr()

        # linear form
        else:
            if interior:
                # could not find any use case
                raise Exception("No interior support in linear facet form.")
            # initialize sparse matrix structures
            data = np.zeros(Nbfun_v*ne)
            rows = np.zeros(Nbfun_v*ne)
            cols = np.zeros(Nbfun_v*ne)

            for i in range(Nbfun_v):
                v1, dv1 = self.elem_v.gbasis(self.mapping, Y1, i, tind1)

                # find correct location in data,rows,cols
                ixs = slice(ne*i, ne*(i + 1))

                # compute entries of local stiffness matrices
                data[ixs] = np.dot(fform(v1, dv1, x, h, n, w)*np.abs(detDG), W)
                rows[ixs] = self.dofnum_v.t_dof[i, tind1]
                cols[ixs] = np.zeros(ne)

            return coo_matrix((data, (rows, cols)),
                              shape=(self.dofnum_v.N, 1)).toarray().T[0]

    def L2error(self, uh, exact, intorder=None):
        """Compute :math:`L^2` error against exact solution.

        The computation is based on the following identity:

        .. math::

            \|u-u_h\|_0^2 = (u,u)+(u_h,u_h)-2(u,u_h).

        Parameters
        ----------
        uh : np.array
            The discrete solution.

        exact : function handle
            The exact solution.

        intorder : (OPTIONAL) int
            The integration order.

        Returns
        -------
        float
            The global :math:`L^2` error.
        """
        if self.elem_u.maxdeg != self.elem_v.maxdeg:
            raise NotImplementedError("elem_u.maxdeg must be elem_v.maxdeg "
                                      "when computing errors!")
        if intorder is None:
            intorder = 2*self.elem_u.maxdeg

        X, W = get_quadrature(self.mesh.refdom, intorder)

        # assemble some helper matrices
        # the idea is to use the identity: (u-uh,u-uh)=(u,u)+(uh,uh)-2(u,uh)
        def uv(u, v):
            return u*v

        def fv(v, x):
            return exact(x)*v

        M = self.iasm(uv)
        f = self.iasm(fv)

        detDF = self.mapping.detDF(X)
        x = self.mapping.F(X)

        uu = np.sum(np.dot(exact(x)**2*np.abs(detDF), W))

        return np.sqrt(uu + np.dot(uh, M.dot(uh)) - 2.*np.dot(uh, f))

    def H1error(self, uh, dexact, intorder=None):
        """Compute :math:`H^1` seminorm error against exact solution.

        The computation is based on the following identity:

        .. math::

            \|\\nabla(u-u_h)\|_0^2 = (\\nabla u,\\nabla u)
                                     + (\\nabla u_h, \\nabla u_h)
                                     - 2(\\nabla u, \\nabla u_h).

        Parameters
        ----------
        uh : np.array
            The discrete solution.

        dexact : function handle
            The derivative of the exact solution.

        intorder : (OPTIONAL) int
            The integration order.

        Returns
        -------
        float
            The global :math:`H^1` error.
        """
        if self.elem_u.maxdeg != self.elem_v.maxdeg:
            raise NotImplementedError("elem_u.maxdeg must be elem_v.maxdeg "
                                      "when computing errors!")
        if intorder is None:
            intorder = 2*self.elem_u.maxdeg

        X, W = get_quadrature(self.mesh.refdom, intorder)

        # assemble some helper matrices
        # the idea is to use the identity: (u-uh,u-uh)=(u,u)+(uh,uh)-2(u,uh)
        def uv(du, dv):
            if not isinstance(du, dict):
                return du*dv
            elif len(du) == 2:
                return du[0]*dv[0] + du[1]*dv[1]
            elif len(du) == 3:
                return du[0]*dv[0] + du[1]*dv[1] + du[2]*dv[2]
            else:
                raise NotImplementedError("AssemblerElement.H1error not "
                                          "implemented for current domain "
                                          "dimension!")

        def fv(dv, x):
            if not isinstance(x, dict):
                return dexact(x)*dv
            elif len(x) == 2:
                return dexact[0](x)*dv[0] + dexact[1](x)*dv[1]
            elif len(x) == 3:
                return dexact[0](x)*dv[0] + dexact[1](x)*dv[1]\
                       + dexact[2](x)*dv[2]
            else:
                raise NotImplementedError("AssemblerElement.H1error not "
                                          "implemented for current domain "
                                          "dimension!")

        M = self.iasm(uv)
        f = self.iasm(fv)

        detDF = self.mapping.detDF(X)
        x = self.mapping.F(X)

        if not isinstance(x, dict):
            uu = np.sum(np.dot((dexact(x)**2) * np.abs(detDF), W))
        elif len(x) == 2:
            uu = np.sum(np.dot((dexact[0](x)**2 + dexact[1](x)**2)
                               * np.abs(detDF), W))
        elif len(x) == 3:
            uu = np.sum(np.dot((dexact[0](x)**2 + dexact[1](x)**2
                                + dexact[2](x)**2) * np.abs(detDF), W))
        else:
            raise NotImplementedError("AssemblerElement.H1error not "
                                      "implemented for current domain "
                                      "dimension!")

        return np.sqrt(uu + np.dot(uh, M.dot(uh)) - 2.*np.dot(uh, f))

class Dofnum(object):
    """Generate a global degree-of-freedom numbering for arbitrary mesh."""

    n_dof = np.array([]) #: Nodal DOFs
    e_dof = np.array([]) #: Edge DOFs (3D only)
    f_dof = np.array([]) #: Facet DOFs (corresponds to edges in 2D)
    i_dof = np.array([]) #: Interior DOFs
    t_dof = np.array([]) #: Global DOFs, number-of-dofs x number-of-triangles
    N = 0 #: Total number of DOFs

    def __init__(self, mesh, element):
        # vertex dofs
        self.n_dof = np.reshape(np.arange(element.n_dofs
                                          * mesh.p.shape[1],
                                          dtype=np.int64),
                                (element.n_dofs, mesh.p.shape[1]), order='F')
        offset = element.n_dofs*mesh.p.shape[1]

        # edge dofs
        if hasattr(mesh, 'edges'): # 3D mesh
            self.e_dof = np.reshape(np.arange(element.e_dofs
                                              * mesh.edges.shape[1],
                                              dtype=np.int64),
                                    (element.e_dofs, mesh.edges.shape[1]),
                                    order='F') + offset
            offset = offset + element.e_dofs*mesh.edges.shape[1]

        # facet dofs
        if hasattr(mesh, 'facets'): # 2D or 3D mesh
            self.f_dof = np.reshape(np.arange(element.f_dofs
                                              * mesh.facets.shape[1],
                                              dtype=np.int64),
                                    (element.f_dofs, mesh.facets.shape[1]),
                                    order='F') + offset
            offset = offset + element.f_dofs*mesh.facets.shape[1]

        # interior dofs
        self.i_dof = np.reshape(np.arange(element.i_dofs
                                          * mesh.t.shape[1],
                                          dtype=np.int64),
                                (element.i_dofs, mesh.t.shape[1]),
                                order='F') + offset

        # global numbering
        self.t_dof = np.zeros((0, mesh.t.shape[1]), dtype=np.int64)

        # nodal dofs
        for itr in range(mesh.t.shape[0]):
            self.t_dof = np.vstack((self.t_dof,
                                    self.n_dof[:, mesh.t[itr, :]]))

        # edge dofs (if 3D)
        if hasattr(mesh, 'edges'):
            for itr in range(mesh.t2e.shape[0]):
                self.t_dof = np.vstack((self.t_dof,
                                        self.e_dof[:, mesh.t2e[itr, :]]))

        # facet dofs (if 2D or 3D)
        if hasattr(mesh, 'facets'):
            for itr in range(mesh.t2f.shape[0]):
                self.t_dof = np.vstack((self.t_dof,
                                        self.f_dof[:, mesh.t2f[itr, :]]))

        self.t_dof = np.vstack((self.t_dof, self.i_dof))

        self.N = np.max(self.t_dof) + 1

    def getdofs(self, N=None, F=None, E=None, T=None):
        """Return global DOF numbers corresponding to each
        node(N), facet(F), edge(E) and triangle(T)."""
        dofs = np.zeros(0, dtype=np.int64)
        if N is not None:
            dofs = np.hstack((dofs, self.n_dof[:, N].flatten()))
        if F is not None:
            dofs = np.hstack((dofs, self.f_dof[:, F].flatten()))
        if E is not None:
            dofs = np.hstack((dofs, self.e_dof[:, E].flatten()))
        if T is not None:
            dofs = np.hstack((dofs, self.i_dof[:, T].flatten()))
        return dofs.flatten()
