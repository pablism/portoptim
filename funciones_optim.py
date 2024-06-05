from tkinter import W
import pandas as pd
import numpy as np
from scipy.optimize import minimize, basinhopping
from itertools import product
from numpy.linalg import cholesky
from mystuff import bdh
import datetime
from numpy.linalg import cholesky, inv
import plotly.graph_objs as go

#optim
def mvo(r,covar,vx,x0,bnds):
    rr =r.values
    def w(x):
        return np.sum(x[:4])-1
    def varp(x):
        return np.dot(np.matmul(x,covar),x)-vx**2
    def rfun(x):
        return -np.dot(x,rr)
    cons  = [{'type': 'eq', 'fun': varp},{'type': 'eq', 'fun': w}]
    res = minimize(rfun,x0,bounds=bnds,constraints=cons)

    return  -rfun(res.x), pd.Series(res.x,index=r.index)



def BLopt(r,sigma,q,c,tau=0.025):
    pick = pick = np.eye(len(r))
    PEP = np.matmul(np.matmul(pick,sigma),pick.T)
    v = r + np.diag(sigma**(1/2))*q # r + 2(pr-0.5)*vol
    omega = (c@PEP)@c
    mu_bl = r + tau*np.matmul(np.matmul(sigma*pick.T,inv(tau*PEP+omega)),v-np.matmul(pick,r))
    sigma_bl = (1+tau)*sigma - tau**2*np.matmul(np.matmul(np.matmul(sigma,pick.T),inv(tau*PEP+omega)),np.matmul(pick,sigma))

    return pd.Series(mu_bl,index=r.index),sigma_bl,v,omega



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


class backt :

    def loadprices(sdate=20191231):
        tk = pd.read_csv('tickers.csv')
        tk = tk.loc[tk['type']=='ASSETCLASS'].set_index('ticker').label.to_dict()
        p = bdh(tk.keys(),'PX_LAST',(sdate,datetime.datetime.today().strftime('%Y%m%d')))
        p.index = pd.to_datetime(p.index)
        p = p.sort_index()
        p.rename(columns=tk,inplace=True)
        p = p.reindex(pd.date_range(p.index[0],p.index[-1])).ffill()

        return p


    def rt(x,p):
        if x.name in ['USDCOP']:
            r = x.pct_change(fill_method=None)+(p.loc[x.index,'USCASH']-p.loc[x.index,'COCASH'])/36500
        elif x.name in ['COCASH','USCASH']:
            r = x.div(100).div(365)
        elif x.name in ['USTTEN']:
            r = -9*x.div(100).diff() + x.div(100).div(365)
        else:
            r = x.pct_change(fill_method=None)
        return r

    def portr(r,sdate,fdate,w,rebal=True,rbw=None):
        dts = pd.date_range(sdate,fdate)
        wp = []
        rp = []
        wp.append(w)
        rp.append(0)
        for i,d in enumerate(dts):
            
            if rebal:
                if d in rbw.index:
                    wp.append(rbw.loc[d,:].values)
                else:
                    wp.append(np.multiply(wp[i],r.loc[d,:].add(1))/(np.dot(wp[i],r.loc[d,:])+1))
            else:
                wp.append(np.multiply(wp[i],r.loc[d,:].add(1))/(np.dot(wp[i],r.loc[d,:])+1))
            rp.append(np.dot(wp[-1],r.loc[d,:]))

        wp = pd.DataFrame(wp[1:],index=dts)
        rp = pd.Series(rp[1:],index=dts)
        return wp, rp
        
    def mdd(x):
        dd = x.copy()
        mx = dd.iloc[0]
        dd.iloc[0] = 0
        for i, r in enumerate(x.index):
            if i > 0:
                if x.iloc[i] >= mx:
                    mx = x.iloc[i]
                    dd.iloc[i] = 0
                else:
                    dd.iloc[i] = x.iloc[i] / mx - 1
        return min(dd)

    def pmetrics(ports,cash,bench=0):
        ret = ports.add(1).apply(np.log).sum().apply(np.exp).subtract(1).mul(100)
        retm = ports.add(1).apply(np.log).resample('M').sum().apply(np.exp).subtract(1).mean().mul(100)
        volm = ports.add(1).apply(np.log).resample('M').sum().apply(np.exp).subtract(1).std().mul(np.sqrt(12)).mul(100)
        # alpha = ret - ports.loc[:,bench].add(1).add(1).apply(np.log).resample('Q').sum().apply(np.exp).subtract(1).mean()
        # alphaq = retq - alpha.resample('Q').mean()
        sharpe = (retm - cash.loc[ports.index].add(1).apply(np.log).resample('M').sum().apply(np.exp).subtract(1).mean()*100)/volm
        b100 = ports.add(1).apply(np.log).cumsum().apply(np.exp).mul(100)
        md = b100.apply(backt.mdd,axis=0).mul(100)
        retms = ports.add(1).apply(np.log).resample('M').sum().apply(np.exp).subtract(1)
        etl  = pd.Series([retms.loc[retms.loc[:,c]<=retms.loc[:,c].quantile(0.05),c].mean() for c in retms.columns],index=ports.columns)
        etg  = pd.Series([retms.loc[retms.loc[:,c]>=retms.loc[:,c].quantile(0.95),c].mean() for c in retms.columns],index=ports.columns)
        etgetl = etg.abs()/etl.abs()
        downdev = pd.Series([np.sqrt((retms.loc[retms.loc[:,c]<=0,c]**2).sum()/len(retms.loc[retms.loc[:,c]<=0,c]))*100 for c in retms.columns],index=ports.columns)
        sortino = retm.div(downdev)
        df = pd.DataFrame({
            'ret':ret,
            'retm':retm,
            'volm':volm,
            'sharpe':sharpe,
            'mdd':md,
            'etg/etl':etgetl,
            'downdev':downdev,
            'sortino':sortino
            },index=ports.columns)
        return df.transpose()
    
class graphs:

    def sct(df,title=''):
        cl = ['black','black','darkblue','darkblue','gray','gray']
        ds = ['solid','dot','solid','dot','solid','dot']
        traces=[]
        for k,v in {i:[c,d] for i,c,d in zip(df.columns,cl,ds)}.items():
            y = df.loc[:,k]
            x = y.index
            traces.append(go.Scatter(x=x,y=y,mode='lines',line={'color':v[0],'width':1,'dash':v[1]},name=k))
    

        layout = go.Layout(width=550,height=400,
                        yaxis={'tickfont':{'size':9}},
                        xaxis={'tickfont':{'size':9}},
                        margin=dict(l=50,r=50,b=50,t=50),
                        xaxis_showgrid=True, yaxis_showgrid=True,
                        showlegend=True,
                        legend={'x':0.05,'y':-0.07,'orientation':'h'},
                        template='simple_white',
                        title={'text':title,'xanchor':'left','font':{'size':12}},
                        )
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def area(df,title='',grouptwo=['']):
        cl = ['rgba(150,120,120,0.7)','rgba(160,160,160,0.7)','rgba(200,200,220,0.7)','rgba(210,230,210,0.7)','rgba(150,185,185,0.6)']


        traces=[]
        for k,v in {i:[c] for i,c in zip(df.columns,cl)}.items():
            
            y = df.loc[:,k]
            x = y.index
    
            if k in grouptwo:
                traces.append(go.Scatter(x=x,y=y,mode='lines',line={'color':'black','width':0.8},name=k))
            else:
                traces.append(go.Scatter(x=x,y=y,mode='lines',line={'color':v[0],'width':1},name=k,stackgroup='one'))
    

        layout = go.Layout(width=550,height=400,
                        yaxis={'tickfont':{'size':9}},
                        xaxis={'tickfont':{'size':9}},
                        yaxis2={'overlaying':'y','side':'right','tickfont':{'size':9}},
                        margin=dict(l=50,r=50,b=50,t=50),
                        xaxis_showgrid=True, yaxis_showgrid=True,
                        showlegend=True,
                        legend={'x':0.05,'y':-0.07,'orientation':'h'},
                        template='simple_white',
                        title={'text':title,'xanchor':'left','font':{'size':12}},
                        )
        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def area_2(df,title='',nofill=['']):
        cl = ['rgba(150,120,120,0.7)','rgba(160,160,160,0.7)','rgba(200,200,220,0.7)','rgba(210,230,210,0.7)','rgba(150,185,185,0.6)']


        traces=[]
        for k,v in {i:[c] for i,c in zip(df.columns,cl)}.items():
            
            y = df.loc[:,k]
            x = y.index
    
            if k in nofill:
                traces.append(go.Scatter(x=x,y=y,mode='lines',line={'color':'black','width':0.8},name=k))
            else:
                traces.append(go.Scatter(x=x,y=y,mode='lines',fillcolor=v[0],line={'color':v[0],'width':1},name=k,fill='tozeroy'))
    

        layout = go.Layout(width=550,height=400,
                        yaxis={'tickfont':{'size':9}},
                        xaxis={'tickfont':{'size':9}},
                        yaxis2={'overlaying':'y','side':'right','tickfont':{'size':9}},
                        margin=dict(l=50,r=50,b=50,t=50),
                        xaxis_showgrid=True, yaxis_showgrid=True,
                        showlegend=True,
                        legend={'x':0.05,'y':-0.07,'orientation':'h'},
                        template='simple_white',
                        title={'text':title,'xanchor':'left','font':{'size':12}},
                        )
        fig = go.Figure(data=traces, layout=layout)
        fig.show()
