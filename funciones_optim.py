import pandas as pd
import numpy as np
from scipy.optimize import minimize, basinhopping
from itertools import product
from numpy.linalg import cholesky

def port_ret(w,r):
    return -np.array(r).dot(w)

def sumauno(w):
    return np.sum(w)-1

def var(w,cov,tv):
    return (np.dot(w.T,np.matmul(cov,w))) - tv**2

def r_gen(n,ret,cor,vol):
    C = cholesky(cor)
    Zc = np.matmul(C,np.random.normal(size=(len(ret),n))).transpose()
    Mu =  np.tile(ret,(n,1))
    S = np.tile(vol,(n,1))
    R = Mu + Zc*S
    return R

def min_vol(bounds,cov,w):
    constraints = ({'type':'eq','fun':lambda x: np.sum(x) -1})
    GAVP = minimize(var,w,args=(cov,0),bounds=bounds,constraints=constraints)
    minv = np.sqrt(var(GAVP.x,cov,0))
    return minv