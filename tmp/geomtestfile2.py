# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:44:11 2015

@author: knl
"""

import numpy as np
import fem.asm
import scipy.sparse.linalg
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import fem.geometry as fegeom
import copy       
        
g=fegeom.GeometryPSLG2D()
g.add_rectangle(marker='outer')
g.add_rectangle(x=0.25,y=0.25,width=0.5,height=0.5,marker='inner')
#g.add_line((0.4,0),(0.4,0.25),marker='tube1',nodes=np.linspace(0,1,20))
#g.add_hole((0.5,0.5))
g.add_circle((0.9,0.1),0.05,marker='circ')
g.add_region((0.5,0.5),1,0.01)
mesh=g.mesh(0.05)
mesh.draw()
plt.hold('on')
mesh.draw_nodes('inner')
mesh.draw_nodes('outer',mark='sr')
#mesh.draw_nodes('tube1',mark='sg')
mesh.draw_nodes('circ',mark='sg')