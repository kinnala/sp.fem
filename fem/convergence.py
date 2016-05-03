"""
A module to simplify creating convergence studies.
Uses *.pkl (pickle) files as key-value-type storage
and enables simple plotting and fitting of linear
functions on logarithmic scale.

@author: Tom Gustafsson


This file is part of sp.fem.

sp.fem is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

sp.fem is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with sp.fem.  If not, see <http://www.gnu.org/licenses/>. 
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

class ConvergencePoint(object):
    pass    

class ConvergenceStudy(object):
    def __init__(self,fname):
        self.fname=fname+".plk"

    def add_point(self,x,y,label='default',data=None):
        # open datastore if exists
        try:
            with open(self.fname,'rb') as fh:
                datastore=pickle.load(fh)
        except IOError:
            datastore={}

        # save point to datastore
        tmp=ConvergencePoint()
        tmp.y=y
        tmp.label=label
        tmp.data=data
        datastore[x]=tmp

        # save datastore to file
        with open(self.fname,'wb') as fh:
            pickle.dump(datastore,fh)

    def plot(self,show_labels=False,exclude_labels=None):
        try:
            with open(self.fname,'rb') as fh:
                datastore=pickle.load(fh)
        except IOError:
            raise Exception("ConvergenceStudy.plot(): File "+self.fname+" not found!")

        graphs_x={}
        graphs_y={}
        for key in datastore:
            pt=datastore[key]
            label=pt.label
            if exclude_labels is not None and label in exclude_labels:
                pass
            else:
                if label in graphs_x:
                    graphs_x[label]=np.append(graphs_x[label],key)
                    graphs_y[label]=np.append(graphs_y[label],pt.y)
                else:
                    graphs_x[label]=np.array([key])
                    graphs_y[label]=np.array([pt.y])

        fig=plt.figure()
        for g in graphs_x:
            plt.loglog(graphs_x[g],graphs_y[g],'bo-')

        return fig

