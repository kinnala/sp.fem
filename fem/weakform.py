# -*- coding: utf-8 -*-
"""
Helper functions to generate weak formulations using SymPy.

@author: Tom Gustafsson
"""
import sympy as s
import numpy as np
import copy
import re

class TensorFunction(object):
    """Wrapper around SymPy for better (more concise) weak form support."""

    def __init__(self,sym='u',dim=1,tdim=1,torder=0):

        if dim>3:
            raise NotImplementedError("TensorFunction.init(): Given dimension not supported!")
        if tdim>3:
            raise NotImplementedError("TensorFunction.init(): Given target dimension not supported!")
        if torder>2:
            raise NotImplementedError("TensorFunction.init(): Given tensorial order not supported!")

        if torder>0 and tdim==1:
            tdim=dim

        self.dim=dim
        self.tdim=tdim
        self.torder=torder
        self.basic_syms=[s.symbols(x) for x in ['x','y','z']]

        def dim_eval(fun,d):
            if d==1:
                return fun(self.basic_syms[0])
            elif d==2:
                return fun(self.basic_syms[0],\
                           self.basic_syms[1])
            elif d==3:
                return fun(self.basic_syms[0],\
                           self.basic_syms[1],\
                           self.basic_syms[2])
            else:
                raise NotImplementedError("TensorFunction.init(): Given domain dimension not supported!")

        if torder==0:
            u=dim_eval(s.Function(sym),dim)
        elif torder==1:
            u={}
            for itr in range(tdim):
                u[itr]=dim_eval(s.Function(sym+str(itr+1)),dim)
        elif torder==2:
            u={}
            for itr in range(tdim):
                u[itr]={}
                for jtr in range(tdim):
                    u[itr][jtr]=dim_eval(s.Function(sym+str(itr+1)+str(jtr+1)),dim)
        else:
            raise NotImplementedError("TensorFunction.init(): Given tensor order not supported!")

        self.expr=u

    def serialize(self):
        stri=''
        if self.torder==0:
            stri=str(self.expr)
        elif self.torder==1:
            for itr in range(self.tdim):
                stri+=str(itr+1)+'. comp.: '+str(self.expr[itr])+'\n'
        elif self.torder==2:
            for itr in range(self.tdim):
                for jtr in range(self.tdim):
                    stri+='('+str(itr+1)+','+str(jtr+1)+') comp.: '+str(self.expr[itr][jtr])+'\n'
        else:
            raise NotImplementedError("TensorFunction.serialize(): Given tensor order not supported!")

        return stri

    def grad(self):
        new=copy.deepcopy(self)
        # compute gradient
        if self.torder==0:
            new.expr={}
            for itr in range(self.dim):
                new.expr[itr]=self.expr.diff(self.basic_syms[itr])
            new.torder=1
            new.tdim=self.dim
        elif self.torder==1:
            for itr in range(self.dim):
                tmp=new.expr[itr]
                new.expr[itr]={}
                for jtr in range(self.dim):
                    new.expr[itr][jtr]=tmp.diff(self.basic_syms[jtr])
                new.torder=2
        else:
            raise NotImplementedError("TensorFunction.grad(): Given tensor order not supported!")
        return new

    def div(self):
        new=copy.deepcopy(self)
        # compute divergence
        if self.torder==1:
            new.expr=0
            for itr in range(self.tdim):
                new.expr+=self.expr[itr].diff(self.basic_syms[itr])
            new.torder=0
        elif self.torder==2:
            new.expr={}
            for itr in range(self.tdim):
                new.expr[itr]=0
                for jtr in range(self.tdim):
                    new.expr[itr]+=self.expr[itr][jtr].diff(self.basic_syms[jtr])
            new.torder=1
        else:
            raise NotImplementedError("TensorFunction.div(): Given tensor order not supported!")
        return new

    def sum(self):
        new=copy.deepcopy(self)
        # sum all entries
        if self.torder==1:
            new.expr=0
            for itr in range(self.tdim):
                new.expr+=self.expr[itr]
            new.torder=0
        elif self.torder==2:
            new.expr=0
            for itr in range(self.tdim):
                for jtr in range(self.tdim):
                    new.expr+=self.expr[itr][jtr]
            new.torder=0
        else:
            raise NotImplementedError("TensorFunction.sum(): Given tensor order not supported!")
        return new

    def T(self):
        new=copy.deepcopy(self)
        if self.torder==2:
            for itr in range(self.tdim):
                for jtr in range(self.tdim):
                    new.expr[itr][jtr]=self.expr[jtr][itr]
        else:
            raise NotImplementedError("TensorFunction.T(): Transpose only supported for torder=2!")
        return new

    def __mul__(self,other):
        new=copy.deepcopy(self)
        if isinstance(other,TensorFunction):
            if self.torder!=other.torder:
                if other.torder==0:
                    other=ConstantTensor(other.expr,dim=new.dim,tdim=new.tdim,torder=new.torder)
                elif self.torder==0:
                    new=ConstantTensor(self.expr,dim=other.dim,tdim=other.tdim,torder=other.torder)
                elif self.torder==2 and other.torder==1:
                    new=ConstantTensor(0.0,dim=self.dim,tdim=self.dim,torder=1)
                    for itr in range(new.tdim):
                        for jtr in range(new.tdim):
                            new.expr[itr]+=self.expr[itr][jtr]*other.expr[jtr]
                    return new
                else:
                    raise Exception("TensorFunction.__mul__(): The given tensors not compatible (different tensorial orders and neither is scalar)!")
            if self.tdim!=other.tdim:
                raise Exception("TensorFunction.__mul__(): The given tensors not compatible (different target dims)!")

        if new.torder==0:
            if isinstance(other,TensorFunction):
                new.expr*=other.expr
            else:
                new.expr*=other
        elif new.torder==1:
            for itr in range(new.tdim):
                if isinstance(other,TensorFunction):
                    new.expr[itr]*=other.expr[itr]
                else:
                    new.expr[itr]*=other
        elif new.torder==2:
            for itr in range(new.tdim):
                for jtr in range(new.tdim):
                    if isinstance(other,TensorFunction):
                        new.expr[itr][jtr]*=other.expr[itr][jtr]
                    else:
                        new.expr[itr][jtr]*=other
        else:
            raise Exception("TensorFunction.__mul__(): The given tensors not compatible (different tensorial orders)!")

        return new

    __rmul__ = __mul__

    def __add__(self,other):
        new=copy.deepcopy(self)
        if isinstance(other,TensorFunction):
            if self.torder!=other.torder:
                raise Exception("TensorFunction.__add__(): The given tensors not compatible (different tensorial orders)!")
            if self.tdim!=other.tdim:
                raise Exception("TensorFunction.__add__(): The given tensors not compatible (different target dims)!")

        if self.torder==0:
            if isinstance(other,TensorFunction):
                new.expr+=other.expr
            else:
                new.expr+=other
        elif self.torder==1:
            for itr in range(self.tdim):
                if isinstance(other,TensorFunction):
                    new.expr[itr]+=other.expr[itr]
                else:
                    new.expr[itr]+=other
        elif self.torder==2:
            for itr in range(self.tdim):
                for jtr in range(self.tdim):
                    if isinstance(other,TensorFunction):
                        new.expr[itr][jtr]+=other.expr[itr][jtr]
                    else:
                        new.expr[itr][jtr]+=other
        else:
            raise Exception("TensorFunction.__mul__(): The given tensors not compatible (different tensorial orders)!")

        return new

    def __repr__(self):
        return self.serialize()

    def __str__(self):
        return self.serialize()

    def __getitem__(self,index):
        if isinstance(index,slice):
            return self.sum().expr
        if isinstance(index,tuple) and len(index)==2:
            if self.torder!=2:
                raise Exception("TensorFunction.__getitem__(): Faulty index, 2-tuple for torder!=2.")
            return self.expr[index[0]][index[1]]
        else:
            try:
                return self.expr[index]
            except:
                raise Exception("TensorFunction.__getitem__(): Cannot index scalar. Use slice ':' instead!")

    def handlify(self,sym1='u',sym2='v',simplify=True,verbose=False):
        if self.torder!=0:
            raise Exception("TensorFunction.handlify(): Tensor must be reduced to scalar (bilinear form) before handlifying!")

        if simplify:
            self.expr=self.expr.simplify()

        wf=self.expr.__str__()

        if "u" in wf:
            bilinear=True
        else:
            bilinear=False

        wf=wf.replace("(x, y, z)","")
        wf=wf.replace("(x, y)","")
        wf=wf.replace("(x)","")

        wf=re.sub(r"("+sym1+r"|"+sym2+r")1","\\1[0]",wf)
        wf=re.sub(r"("+sym1+r"|"+sym2+r")2","\\1[1]",wf)
        wf=re.sub(r"("+sym1+r"|"+sym2+r")3","\\1[2]",wf)

        wf=re.sub(r"Derivative\((("+sym1+r"|"+sym2+r")(\[\d\])?), x\)","d\\1[0]",wf)
        wf=re.sub(r"Derivative\((("+sym1+r"|"+sym2+r")(\[\d\])?), y\)","d\\1[1]",wf)
        wf=re.sub(r"Derivative\((("+sym1+r"|"+sym2+r")(\[\d\])?), z\)","d\\1[2]",wf)

        wf=wf.replace("x","x[0]")
        wf=wf.replace("y","x[1]")
        wf=wf.replace("z","x[2]")

        wf=wf.replace("sin","np.sin")
        wf=wf.replace("cos","np.cos")

        wf=wf.replace("pi","np.pi")

        if verbose:
            print wf

        if bilinear:
            return eval("lambda u,v,du,dv,x: "+wf)
        else:
            return eval("lambda v,dv,x: "+wf)

class IdentityMatrix(TensorFunction):
    def __init__(self,d):
        TensorFunction.__init__(self,dim=d,torder=2)
        for itr in range(self.tdim):
            for jtr in range(self.tdim):
                if itr==jtr:
                    self.expr[itr][jtr]=1.0
                else:
                    self.expr[itr][jtr]=0.0

class ConstantTensor(TensorFunction):
    def __init__(self,const,dim=1,tdim=1,torder=0):
        TensorFunction.__init__(self,dim=dim,tdim=tdim,torder=torder)
        if torder==0:
            self.expr=const
        elif torder==1:
            for itr in range(self.tdim):
                self.expr[itr]=const
        elif torder==2:
            for itr in range(self.tdim):
                for jtr in range(self.tdim):
                    self.expr[itr][jtr]=const
        else:
            raise NotImplementedError("ConstantTensor.__init__(): Not implemented for torder>2!")

class BasisTensor(TensorFunction):
    def __init__(self,i,j=None,dim=2):
        if j is None:
            TensorFunction.__init__(self,dim=dim,torder=1)
        else:
            TensorFunction.__init__(self,dim=dim,torder=2)

        if j is None:
            for itr in range(self.tdim):
                if i==itr:
                    self.expr[itr]=1.0
                else:
                    self.expr[itr]=0.0
        else:
            for itr in range(self.tdim):
                for jtr in range(self.tdim):
                    if i==itr and j==jtr:
                        self.expr[itr][jtr]=1.0
                    else:
                        self.expr[itr][jtr]=0.0

def div(W):
    return W.div()

def grad(W):
    return W.grad()

def dotp(A,B):
    return (A*B).sum()
