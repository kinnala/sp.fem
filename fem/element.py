class Element:
  """Superclass for all finite elements."""

  maxdeg=0 # maximum polynomial degree; for determining quadrature
  dim=0 # spatial dimension
  tdim=0 # target dimension
  torder=0 # target tensorial order

  """
  The discretized field is a mapping
     U : C^dim -> C^(tdim x ... x tdim)
  where the product is taken 'torder' times.
  """

  # number of ...
  n_dofs=0 # nodal dofs
  i_dofs=0 # interior dofs
  f_dofs=0 # facet dofs (2d and 3d only)
  e_dofs=0 # edge dofs (3d only)

  def lbasis(self,X,i):
    """
    Returns local basis functions
    evaluated at some local points.
    """
    raise NotImplementedError("Element local basis (lbasis) not implemented!")

  def gbasis(self,X,i):
    """
    Returns global basis functions
    evaluated at some local points.
    """
    raise NotImplementedError("Element local basis (lbasis) not implemented!")

class ElementH1(Element):
  """Superclass for H1 conforming finite elements."""

  def gbasis(self,X,i):
    [phi,dphi]=self.lbasis(X,i)
