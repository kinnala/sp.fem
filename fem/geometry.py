import numpy as np
import fem.mesh
import platform

import os
import matplotlib.pyplot as plt

class Geometry:
    """Superclass for all geometries."""

    def __init__(self):
        raise NotImplementedError("Geometry constructor not implemented!")

    def mesh(self):
        raise NotImplementedError("Geometry mesher not implemented!")

class GeometryPSLG2D(Geometry):
    """A geometry defined by PSLG (planar straight line graph).

    PSLG's can be meshed using Triangle.
    
    This class also contains some methods to help defining
    two-dimensional geometry boundaries."""

    def __init__(self):
        self.segments=[]
        # each segment is a dictionary with the following keys
        #   vertices = 2xN list of points in proper order
        #   marker = string, name for the segment, for
        #            identifying nodes belonging to it
        self.holes=[]
        # each hole is a 2-tuple with x- and y-coordinates of hole
        self.regions=[]
        # each region is a 4-tuple (x,y,number,areaconstraint)
        # each region is a dictionary with the following keys
        #   location = 2-tuple with x- and y-coordinates
        #   maxh = maximum mesh parameter for the area.
        #          (-1 == no constraint)
        #   marker = string, name for the region. identifies
        #            triangles belonging to it.
        self.regionid={}

    def add_segment(self,vertices,marker=None):
        """Create a new boundary segment and append it to segment list."""
        if marker==None:
            marker='boundary'
        newseg={}
        newseg['vertices']=vertices
        newseg['marker']=marker
        self.segments.append(newseg)

    def add_hole(self,location):
        """Add a hole."""
        self.holes.append(location)

    def add_region(self,location,marker=None,h=-1):
        """Add a region."""
        if marker==None:
            marker='other'
        newreg={}
        newreg['location']=location
        newreg['maxh']=h
        newreg['marker']=marker
        self.regionid[marker]=len(self.regions)+1
        self.regions.append(newreg)

    def add_line(self,p1,p2,marker=None,nodes=np.array([0,1])):
        """Add a line.
        
        Include optional np.array 'nodes'
        with parameters in the interval [0,1] to include
        other than endpoint nodes.
        
        e.g. add_line((0,0),(1,1),nodes=np.linspace(0,1,10))"""
        xs=nodes*p1[0]+(1-nodes)*p2[0]
        ys=nodes*p1[1]+(1-nodes)*p2[1]
        self.add_segment(np.vstack((xs,ys)),marker=marker)

    def add_rectangle(self,x=0,y=0,width=1,height=1,marker=None):
        """Add a rectangle."""
        self.add_segment(np.array([[x,y],[x+width,y],[x+width,y+height],[x,y+height],[x,y]]).T,marker=marker)

    def add_circle(self,c=(0.0,0.0),r=1.0,nodes=np.linspace(0,2*np.pi,11),marker=None):
        """Add a circle.
        
        Include optional np.array 'nodes'
        with parameters in the interval [0,2*pi] to include
        more than 10 nodes.
        
        e.g. add_circle((0,0),1.0,nodes=np.linspace(0,2*np.pi,50))"""
        xs=r*np.cos(nodes)+c[0]
        ys=r*np.sin(nodes)+c[1]
        self.add_segment(np.vstack((xs,ys)),marker=marker)

    def draw(self,markers=False,regions=False,holes=False):
        """Draw the segments.

        Optionally visualize points with region marks
        and hole marks."""
        fig=plt.figure()
        # visualize the geometry
        xs=[]
        ys=[]
        for seg in self.segments:
            for jtr in range(seg['vertices'].shape[1]-1):
                xs.append(seg['vertices'][0,jtr])
                xs.append(seg['vertices'][0,jtr+1])
                xs.append(None)
                ys.append(seg['vertices'][1,jtr])
                ys.append(seg['vertices'][1,jtr+1])
                ys.append(None)
        plt.plot(xs,ys,'k')
        if markers:
            for itr in self.segments:
                plt.text(itr['vertices'][0,0],\
                        itr['vertices'][1,0],\
                        itr['marker'],\
                        bbox=dict(facecolor='green', alpha=0.8))
        if regions:
            for itr in self.regions:
                plt.text(itr['location'][0],\
                        itr['location'][1],\
                        itr['marker'],\
                        bbox=dict(facecolor='red', alpha=0.8))
        if holes:
            for itr in self.holes:
                plt.plot(itr[0],itr[1],'rx')
        return fig

    def mesh(self,hmax=1.0,minangle=20.0):
        """Mesh the defined geometry using Triangle."""
        commfile="geom"
        # total number of vertices from self.segments
        N=np.sum(np.array([seg['vertices'].shape[1] for seg in self.segments]))
        # total number of segments
        M=len(self.segments)
        # total number of 'Triangle segments'
        P=np.sum(np.array([seg['vertices'].shape[1]-1 for seg in self.segments]))

        # enumerate boundary markers
        markeri=1
        self.markers={}
        for itr in self.segments:
            if itr['marker'] not in self.markers:
                self.markers[itr['marker']]=markeri
                markeri=markeri+1

        # check that there is at least one segment
        if M==0:
            raise Exception(self.__class__.__name__+": Cannot generate mesh since no boundary segments are defined!")

        # write the boundary segments to Triangle input format
        try:
            f=open(commfile+".poly",'w') 
            # format: # of vertices | dimension | # of attributes | # of markers
            f.write('%d 2 0 0\n'%N)

            offset=0
            # loop over segments
            for itr in range(0,M):
                # loop over vertices of segment
                for jtr in range(0,self.segments[itr]['vertices'].shape[1]):
                    # format: vertex # | x | y
                    f.write('%d %f %f\n'%(jtr+offset,self.segments[itr]['vertices'][0,jtr],self.segments[itr]['vertices'][1,jtr]))
                offset+=self.segments[itr]['vertices'].shape[1]
            # format: # of segments | # of boundary markers
            f.write('%d 1\n'%P)
            segmenti=0
            offset=0
            for itr in range(0,M):
                for jtr in range(0,self.segments[itr]['vertices'].shape[1]-1):
                    f.write('%d %d %d %s\n'%(segmenti,jtr+offset,jtr+1+offset,self.markers[self.segments[itr]['marker']]))
                    segmenti=segmenti+1
                offset+=self.segments[itr]['vertices'].shape[1]

            # write hole markers
            if len(self.holes)==0:
                f.write('0\n')
            else:
                f.write('%d\n'%len(self.holes))
                for itr in range(0,len(self.holes)):
                    f.write('%d %f %f\n'%(itr,self.holes[itr][0],self.holes[itr][1]))

            # regional attributes
            if len(self.regions)==0:
                f.write('0')
            else:
                f.write('%d\n'%len(self.regions))
                for itr in range(0,len(self.regions)):
                    if self.regions[itr]['maxh']==-1:
                        areaconstr=-1
                    else:
                        areaconstr=self.regions[itr]['maxh']**2
                    f.write('%d %f %f %d %f\n'\
                            %(itr,\
                            self.regions[itr]['location'][0],\
                            self.regions[itr]['location'][1],\
                            self.regionid[self.regions[itr]['marker']],\
                            areaconstr))

            f.close()
        except:
            raise Exception(self.__class__.__name__+": Error when writing Triangle input file!")

        # call Triangle to mesh the domain
        if len(self.regions)==0:
            self.call_triangle("-q%f -Q -a%f -p"%(minangle,hmax**2),inputfile=commfile)
        else:
            self.call_triangle("-q%f -Q -a -A -p"%minangle,inputfile=commfile)

        # load output of Triangle
        mesh=self.load_triangle_mesh(inputfile=commfile)

        return mesh

    def call_triangle(self,args,inputfile="geom"):
        # run Triangle (OS dependent)
        if platform.system()=="Linux":
            os.system("./fem/triangle/triangle "+args+" "+inputfile+".poly")
        elif platform.system()=="Windows":
            os.system("fem\\triangle\\triangle.exe "+args+" "+inputfile+".poly")
        else:
            raise NotImplementedError(self.__class__.__name__+": method 'call_triangle' not implemented for your platform!")

    def load_triangle_mesh(self,inputfile="geom",keepfiles=False):
        try:
            t=np.loadtxt(open(inputfile+".1.ele","rb"),delimiter=None,comments="#",skiprows=1).T
            p=np.loadtxt(open(inputfile+".1.node","rb"),delimiter=None,comments="#",skiprows=1).T
        except:
            raise Exception(self.__class__.__name__+": A problem when loading Triangle output files!")

        if not keepfiles:
            try:
                os.remove(inputfile+".poly")
                os.remove(inputfile+".1.ele")
                os.remove(inputfile+".1.node")
                os.remove(inputfile+".1.poly")
            except:
                print self.__class__.__name__+": (WARNING) Error when removing Triangle output files!"

        # extract index sets corresponding to boundary markers
        markers=p[3,:]
        indexsets=self.markers.copy()

        for key,value in self.markers.iteritems():
            indexsets[key]=np.nonzero(markers==value)[0]

        # extract index sets corresponding to regions (if exist)
        if t.shape[0]==5:
            tmarkers=t[4,:]
            tindexsets=self.regionid.copy()

            for key,value in self.regionid.iteritems():
                tindexsets[key]=np.nonzero(tmarkers==value)[0]
        else:
            tindexsets=None


        return fem.mesh.MeshTri(p[1:3,:],t[1:4,:].astype(np.intp),fixmesh=True,markers=indexsets,tmarkers=tindexsets)

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
        e=mesh.edges
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
    """A geometry defined by a triangular mesh in COMSOL *.mphtext format."""

    p=np.empty([2,0],dtype=np.float_)
    t=np.empty([3,0],dtype=np.intp)

    def __init__(self,filename):
        # TODO make this multiplatform
        if platform.system()!="Linux":
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
