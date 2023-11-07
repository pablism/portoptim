from tkinter import W
import pandas as pd
import numpy as np
from scipy.optimize import minimize, basinhopping
from itertools import product
from numpy.linalg import cholesky
from mystuff import bdh

#optim

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


#other

def normi(x,r):
    m = x.rolling(r,min_periods=1).mean()
    s = x.rolling(r,min_periods=1).std()
    z = x.subtract(m).div(s)
    return z

def momm(x,mas,mal):
    z = x.rolling(mas,min_periods=1).sum()
    s = z.rolling(mal).std()

    return z.div(s)

def prat(pr,wds):
    ratios = []
    for wd in wds:
        base_ = pr.rolling(wd).max() - pr.rolling(wd).min()
        level = pr - pr.rolling(wd).min()
        x = level.div(base_)
        ratios.append(x)
    ratios = pd.concat(ratios,axis=1)
    ratios.columns = [f'{pr.name}_{i}' for i in wds]
    return ratios

class _data:
    def __init__(self):
        refs = pd.read_csv('tickers.csv',index_col='ticker')
        ticks = refs.label.to_dict()
        data = bdh(ticks.keys(),'px_last',(20101231,20231026))
        data.columns = pd.MultiIndex.from_tuples([(refs.loc[c,'label'],refs.loc[c,'type']) for c in data.columns],names=('label','typ'))
        data.index = pd.to_datetime(data.index)
        dataw = data.resample('W-Wed').last().ffill().dropna(axis=0)
        x_ = dataw.loc[:,[c[1] in ['MACRO','SENTIMENT','POSITIONING','MARKETIMPLIED'] for c in dataw.columns]].droplevel(level=1,axis=1)
        x_['AAII'] = x_.BULL.subtract(x_.BEAR)
        x_.drop(['BULL','BEAR'],axis=1,inplace=True)
        x_ = x_.apply(normi,axis=0,args=[260])
        rtw_ = dataw.xs('ASSETCLASS',level='typ',axis=1).drop(['USCASH','COCASH'],axis=1).dropna(axis=0)

        rtw = rtw_.pct_change()
        rtw.USTTEN = rtw_.USTTEN.diff()
        
        pt_ = data.xs('ASSETCLASS',level='typ',axis=1).drop(['USCASH','COCASH'],axis=1).dropna(axis=0)
        rtd = pt_.pct_change()
        rtd.USTTEN = pt_.USTTEN.diff()
        prat_ = pd.concat([prat(pt_.loc[:,c],[4,8,16,32,64]) for c in pt_.columns],axis=1).dropna(axis=0)

        self.data = data
        self.macrox = x_
        self.prat = prat_ 
        self.rtw = rtw
        self.rtd = rtd

    def set_labels(self,w):
        def signal(rtw,w):
            y = rtw.rolling(w).sum()
            y.loc[y>0] = 1
            y.loc[y<0] = 0
            y = y.shift(-w)
            return y
        self.ylabels = self.rtd.apply(signal,args=[w],axis=0).dropna(axis=0)
