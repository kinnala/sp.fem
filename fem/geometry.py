import numpy as np
import fem.mesh

import os
import matplotlib.pyplot as plt
import shapely.geometry as shgeom

class Geometry:
    """
    Superclass for all geometries.
    """

    def __init__(self):
        raise NotImplementedError("Geometry constructor not implemented!")

    def mesh(self):
        raise NotImplementedError("Geometry mesher not implemented!")

class GeometryShapelyTriangle2D(Geometry):
    """
    Shapely geometry meshed using Triangle.

    Geometry tuples are added to list and then
    processed with Shapely. The resulting exterior
    curve is fed to Triangle.

    Tuple examples:
      ('+','circle',<center x>,<center y>,<radius>,<resolution>)
      ('-','polygon',<list of points as 2xN np.array>)
      ('+','box',<min x>,<min y>,<max x>,<max y>)

    Resolution is a natural number. Higher number gives more
    refined boundary edges.
    """

    def __init__(self,glist):
        self.glist=glist

    def process(self):
        if self.glist[0][0]!='+':
            raise Exception("GeometryShapelyTriangle2D: first geometry tuple must be ('+',...)!")
        self.g=self.resolve_gtuple(self.glist[0])
        iterg=iter(self.glist)
        next(iterg)
        for itr in iterg:
            if itr[0] is '+':
                self.g=self.g.union(self.resolve_gtuple(itr))
            elif itr[0] is '-':
                self.g=self.g.difference(self.resolve_gtuple(itr))
            else:
                raise Exception("GeometryShapelyTriangle2D: first item in gtuple must be '+' or '-'!")


    def resolve_gtuple(self,gtuple):
        if gtuple[1] is 'circle':
            return shgeom.Point(gtuple[2],gtuple[3]).buffer(gtuple[4],gtuple[5])
        elif gtuple[1] is 'polygon':
            return shgeom.Polygon([tuple(i) for i in gtuple[2].T])
        elif gtuple[1] is 'box':
            return shgeom.box(gtuple[2],gtuple[3],gtuple[4],gtuple[5])
        else:
            raise NotImplementedError("GeometryShapelyTriangle2D.resolve_gtuple: given shape not implemented!")

    def draw(self):
        """
        Draw the boundary curves of the geometric object.

        Run matplotlib plt.show() after this.
        """
        if isinstance(self.g.boundary,shgeom.multilinestring.MultiLineString):
            # iterate over boundaries
            plt.figure()
            for itr in self.g.boundary:
                xs=np.array([])
                ys=np.array([])
                for jtr in itr.coords:
                    xs=np.append(xs,jtr[0])
                    ys=np.append(ys,jtr[1])
                plt.plot(xs,ys,'k-')
        else:
            # convert to numpy arrays and plot
            xs=np.array([])
            ys=np.array([])
            for itr in self.g.boundary.coords:
                xs=np.append(xs,itr[0])
                ys=np.append(ys,itr[1])
            plt.plot(xs,ys,'k-')

    def mesh(self,hmax=1.0):
        """
        Call triangle to generate a mesh.
        """
        xs=np.array([])
        ys=np.array([])
        segstart=np.array([])
        segend=np.array([])
        itrn=0
        if isinstance(self.g.boundary,shgeom.multilinestring.MultiLineString):
            # iterate over boundaries
            for itr in self.g.boundary:
                for jtr in itr.coords:
                    xs=np.append(xs,jtr[0])
                    ys=np.append(ys,jtr[1])
                    segstart=np.append(segstart,itrn)
                    segend=np.append(segend,itrn+1)
                    itrn=itrn+1
                segstart=segstart[0:-1]
                segend=segend[0:-1]
        else:
            for itr in self.g.boundary.coords:
                xs=np.append(xs,itr[0])
                ys=np.append(ys,itr[1])
                segstart=np.append(segstart,itrn)
                segend=np.append(segend,itrn+1)
                itrn=itrn+1

        f=open('geom.poly','w') 
        f.write('%d 2 0 0\n'%len(xs))
        for itr in range(0,len(xs)):
            f.write('%d %f %f\n'%(itr,xs[itr],ys[itr]))
        f.write('%d 0\n'%len(segstart))
        for itr in range(0,len(segstart)):
            f.write('%d %d %d\n'%(itr,segstart[itr],segend[itr]))
        f.write('0\n')
        f.write('0')
        f.close()

        os.system("./triangle/triangle -q -a%f -p geom.poly > /dev/null"%hmax**2)

        mesh=self.load_triangle("geom")

        os.system("rm geom.poly")
        os.system("rm geom.1.ele")
        os.system("rm geom.1.node")
        os.system("rm geom.1.poly")

        return mesh

    def load_triangle(self,fname):
        t=np.loadtxt(open(fname+".1.ele","rb"),delimiter=None,comments="#",skiprows=1).T
        p=np.loadtxt(open(fname+".1.node","rb"),delimiter=None,comments="#",skiprows=1).T
        return fem.mesh.MeshTri(p[1:3,:],t[1:,:].astype(np.intp))


class GeometryMeshTri(Geometry):
    """
    A geometry defined by a triangular mesh.
    """

    p=np.empty([2,0],dtype=np.float_)
    t=np.empty([3,0],dtype=np.intp)

    def __init__(self,p=np.array([[0,1,0,1],[0,0,1,1]],dtype=np.float_),t=np.array([[0,1,2],[1,2,3]],dtype=np.intp).T):
        self.p=p
        self.t=t

    def mesh(self):
        return fem.mesh.MeshTri(self.p,self.t)

    def refine(self,N=1):
        """
        Perform one or more refines on the mesh.
        """
        for itr in range(N):
            self.single_refine()

    def single_refine(self):
        """
        Perform a single mesh refine.
        """
        mesh=fem.mesh.MeshTri(self.p,self.t)
        # rename variables
        t=mesh.t
        p=mesh.p
        e=mesh.facets
        t2f=mesh.t2f
        # new vertices are the midpoints of edges
        newp=0.5*np.vstack((p[0,e[0,:]]+p[0,e[1,:]],p[1,e[0,:]]+p[1,e[1,:]]))
        newp=np.hstack((p,newp))
        # build new triangle definitions
        sz=p.shape[1]
        newt=np.vstack((t[0,:],t2f[0,:]+sz,t2f[2,:]+sz))
        newt=np.hstack((newt,np.vstack((t[1,:],t2f[0,:]+sz,t2f[1,:]+sz))))
        newt=np.hstack((newt,np.vstack((t[2,:],t2f[2,:]+sz,t2f[1,:]+sz))))
        newt=np.hstack((newt,np.vstack((t2f[0,:]+sz,t2f[1,:]+sz,t2f[2,:]+sz))))
        # update fields
        self.p=newp
        self.t=newt

