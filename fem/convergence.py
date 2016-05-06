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

    def add_point(self,x,y,tag='default'):
        # open datastore if exists
        try:
            with open(self.fname,'rb') as fh:
                datastore=pickle.load(fh)
        except IOError:
            datastore={}

        # save point to datastore
        datastore[(tag,x)]=y

        # save datastore to file
        with open(self.fname,'wb') as fh:
            pickle.dump(datastore,fh)

    def plot(self,xlabel='Mesh parameter',ylabel='Error',
             show_labels=False,loc='upper right',exclude_tags=None,draw_fit=True):
        try:
            with open(self.fname,'rb') as fh:
                datastore=pickle.load(fh)
        except IOError:
            raise Exception("ConvergenceStudy.plot(): File "+self.fname+" not found!")

        graphs_x={}
        graphs_y={}
        for key in datastore:
            pt=datastore[key]
            tag=key[0]
            if exclude_tags is not None and tag in exclude_tags:
                pass
            else:
                if tag in graphs_x:
                    graphs_x[tag]=np.append(graphs_x[tag],key[1])
                    graphs_y[tag]=np.append(graphs_y[tag],pt)
                else:
                    graphs_x[tag]=np.array([key[1]])
                    graphs_y[tag]=np.array([pt])

        fig,ax=plt.subplots()
        for tag in graphs_x:
            I=np.argsort(graphs_x[tag])
            ax.loglog(graphs_x[tag][I],graphs_y[tag][I],'o',
                      label=tag)
            if draw_fit:
                fitcoeffs=np.polyfit(np.log10(graphs_x[tag]),np.log10(graphs_y[tag]),1)
                def fitmap(x):
                    return 10.0**(fitcoeffs[0]*np.log10(x)+fitcoeffs[1])
                def default_fit_label(tag,rate):
                    ratestr='%.2f'%round(rate,2)
                    return "polynomial fit ("+tag+"), slope: "+ratestr
                pts=np.array([graphs_x[tag][I[0]],graphs_x[tag][I[-1]]])
                ax.loglog(pts,fitmap(pts),'-',label=default_fit_label(tag,fitcoeffs[0]))

        if show_labels:
            ax.legend(loc=loc)

        ax.grid(b=True,which='major',color='k',linestyle='-')
        ax.grid(b=True,which='minor',color='0.5',linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig,ax

    def show(self):
        plt.show()

