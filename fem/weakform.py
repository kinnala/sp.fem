# -*- coding: utf-8 -*-
"""
Helper functions to generate weak formulations using SymPy.

@author: Tom Gustafsson
"""
import sympy as s
import copy
import re

class BilinearForm(object):
    pass

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
                raise Exception("TensorFunction.__mul__(): The given tensors not compatible (different tensorial orders)!")
            if self.tdim!=other.tdim:
                raise Exception("TensorFunction.__mul__(): The given tensors not compatible (different target dims)!")

        if self.torder==0:
            if isinstance(other,TensorFunction):
                new.expr*=other.expr
            else:
                new.expr*=other
        elif self.torder==1:
            for itr in range(self.tdim):
                if isinstance(other,TensorFunction):
                    new.expr[itr]*=other.expr[itr]
                else:
                    new.expr[itr]*=other
        elif self.torder==2:
            for itr in range(self.tdim):
                for jtr in range(self.tdim):
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

    def handlify(self,sym1='u',sym2='v'):
        if self.torder!=0:
            raise Exception("TensorFunction.handlify(): Tensor must be reduced to scalar (bilinear form) before handlifying!")

        wf=self.expr.__str__()

        wf=wf.replace("(x, y, z)","")
        wf=wf.replace("(x, y)","")

        wf=re.sub(r"("+sym1+r"|"+sym2+r")1","\\1[0]",wf)
        wf=re.sub(r"("+sym1+r"|"+sym2+r")2","\\1[1]",wf)
        wf=re.sub(r"("+sym1+r"|"+sym2+r")3","\\1[2]",wf)

        wf=re.sub(r"Derivative\((("+sym1+r"|"+sym2+r")(\[\d\])?), x\)","d\\1[0]",wf)
        wf=re.sub(r"Derivative\((("+sym1+r"|"+sym2+r")(\[\d\])?), y\)","d\\1[1]",wf)
        wf=re.sub(r"Derivative\((("+sym1+r"|"+sym2+r")(\[\d\])?), z\)","d\\1[2]",wf)

        print wf

