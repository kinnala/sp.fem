import numpy as np
import fem.mesh
import platform

import os
import matplotlib.pyplot as plt
try:
    import shapely.geometry as shgeom
    opt_shapely=True
except:
    opt_shapely=False

class Geometry:
    """Superclass for all geometries."""

    def __init__(self):
        raise NotImplementedError("Geometry constructor not implemented!")

    def mesh(self):
        raise NotImplementedError("Geometry mesher not implemented!")

class GeometryShapelyTriangle2D(Geometry):
    """Shapely geometry meshed using Triangle.

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
        if opt_shapely:
            self.glist=glist
        else:
            raise ImportError("Shapely not supported by the host system!")

    def process(self):
        if self.glist[0][0]!='+':
            raise Exception("GeometryShapelyTriangle2D: first geometry tuple must be ('+',...)!")
        self.g=self.resolve_gtuple(self.glist[0])
        iterg=iter(self.glist)
        next(iterg)
        for itr in iterg:
            if itr[0]=='+':
                self.g=self.g.union(self.resolve_gtuple(itr))
            elif itr[0]=='-':
                self.g=self.g.difference(self.resolve_gtuple(itr))
            else:
                raise Exception("GeometryShapelyTriangle2D: first item in gtuple must be '+' or '-'!")


    def resolve_gtuple(self,gtuple):
        if gtuple[1]=='circle':
            return shgeom.Point(gtuple[2],gtuple[3]).buffer(gtuple[4],gtuple[5])
        elif gtuple[1]=='polygon':
            return shgeom.Polygon([tuple(i) for i in gtuple[2].T])
        elif gtuple[1]=='box':
            return shgeom.box(gtuple[2],gtuple[3],gtuple[4],gtuple[5])
        else:
            raise NotImplementedError("GeometryShapelyTriangle2D.resolve_gtuple: given shape not implemented!")

    def draw(self):
        """Draw the boundary curves of the geometric object.

        Run matplotlib plt.show() after this.
        """
        # if there is multiple boundaries
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

    def mesh(self,hmax=1.0,minangle=20.0,holes=None):
        """Call Triangle to generate a mesh."""
        # TODO fix meshing of holes
        # process the geometry list with Shapely
        self.process()
        # data arrays for boundary line segments
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

        # write the boundary segments to Triangle input format
        f=open('geom.poly','w') 
        f.write('%d 2 0 0\n'%len(xs))
        for itr in range(0,len(xs)):
            f.write('%d %f %f\n'%(itr,xs[itr],ys[itr]))
        f.write('%d 0\n'%len(segstart))
        for itr in range(0,len(segstart)):
            f.write('%d %d %d\n'%(itr,segstart[itr],segend[itr]))

        # write hole markers
        if holes is None:
            f.write('0\n')
        else:
            f.write('%d\n'%len(holes))
            itrn=0
            for itr in holes:
                f.write('%d %f %f\n'%(itrn,itr[0],itr[1]))
                itrn=itrn+1

        # TODO implement regional attributes
        f.write('0')
        f.close()

        # run Triangle (OS dependent)
        if platform.system()=="Linux":
            os.system("./fem/triangle/triangle -q%f -Q -a%f -p geom.poly"%(minangle,hmax**2))
        elif platform.system()=="Windows":
            os.system("fem\\triangle\\triangle.exe -q%f -Q -a%f -p geom.poly"%(minangle,hmax**2))
        else:
            raise NotImplementedError("GeometryShapelyTriangle2D: Not implemented for your platform!")

        # load output of Triangle
        mesh=self.load_triangle("geom")

        if platform.system()=="Linux":
            os.system("rm geom.poly")
            os.system("rm geom.1.ele")
            os.system("rm geom.1.node")
            os.system("rm geom.1.poly")

        return mesh

    def load_triangle(self,fname):
        try:
            t=np.loadtxt(open(fname+".1.ele","rb"),delimiter=None,comments="#",skiprows=1).T
            p=np.loadtxt(open(fname+".1.node","rb"),delimiter=None,comments="#",skiprows=1).T
        except:
            raise Exception("GeometryShapelyTriangle2D: A problem with meshing!")
        return fem.mesh.MeshTri(p[1:3,:],t[1:,:].astype(np.intp),fixmesh=True)

class GeometryMeshTri(Geometry):
    """A geometry defined by a triangular mesh."""

    p=np.empty([2,0],dtype=np.float_)
    t=np.empty([3,0],dtype=np.intp)

    def __init__(self,p=np.array([[0,1,0,1],[0,0,1,1]],dtype=np.float_),t=np.array([[0,1,2],[1,2,3]],dtype=np.intp).T):
        self.p=p
        self.t=t

    def mesh(self):
        return fem.mesh.MeshTri(self.p,self.t)

    def refine(self,N=1):
        """Perform one or more refines on the mesh."""
        for itr in range(N):
            self.single_refine()

    def single_refine(self):
        """Perform a single mesh refine."""
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


class GeometryMeshTriComsol(GeometryMeshTri):
    """A geometry defined by a mesh in COMSOL *.mphtext format."""

    p=np.empty([2,0],dtype=np.float_)
    t=np.empty([3,0],dtype=np.intp)

    def __init__(self,filename):
        if platform.system()=="Windows":
            raise NotImplementedError("GeometryMeshTriComsol: Loading Comsol meshes not supported on this platform!")
        os.system("csplit "+filename+" '/^# Mesh point coordinates/' > /dev/null")
        os.system("mv xx01 tmp.fem")
        os.system("csplit tmp.fem '/^# Elements/' > /dev/null")
        os.system("mv xx00 vertices.fem")
        os.system("mv xx01 elements.fem")
        os.system("csplit elements.fem '/^$/' > /dev/null")
        os.system("rm elements.fem")
        os.system("mv xx00 elements.fem")
        os.system("csplit vertices.fem '/^$/' > /dev/null")
        os.system("rm vertices.fem")
        os.system("mv xx00 vertices.fem")
        t=np.loadtxt('elements.fem',dtype=np.int64).T
        p=np.loadtxt('vertices.fem').T
        self.p=p
        self.t=t
