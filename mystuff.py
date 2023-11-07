import blpapi
import pandas as pd
import numpy as np

def bdh(tickers,fld,dts):
    #tickers must be a list
    #fld is a String with bloomberg field (only 1 fld at a time is accepted)
    #dts is a tuple with start and end date
    sessionOptions = blpapi.SessionOptions()
    session = blpapi.Session(sessionOptions)
    session.start()
    session.openService("//blp/refdata")
    refDataService = session.getService("//blp/refdata")
    d1 = dts[0]
    d2 = dts[1]
    request = refDataService.createRequest("HistoricalDataRequest")
    for t in tickers:
        request.getElement('securities').appendValue(t)
    request.getElement("fields").appendValue(fld)
    request.set("startDate", str(d1))
    request.set("endDate", str(d2))
    request.set("maxDataPoints", 100000)
    session.sendRequest(request)
    df=[]
    while(True):
        ev = session.nextEvent(500)
        for msg in ev:
            if msg.messageType() == 'HistoricalDataResponse':
                sec = msg.getElement('securityData').getElementAsString('security')
                for s in msg.asElement().getChoice().elements():
                    if s.name() =='fieldData':
                        dt=[]
                        vl=[]
                        for i in s.values():
                            #print(i.getElementAsFloat('PX_LAST'))
                            vl.append(i.getElementAsFloat(fld))
                            dt.append(i.getElementAsDatetime('date'))
                            #print(i.getElementAsDatetime('date'))
                        df.append(pd.DataFrame([dt,vl],index=['date',sec]).transpose().set_index('date'))

        if ev.eventType() == blpapi.Event.RESPONSE:
                break
    session.stop()
    df = pd.concat(df,axis=1)
    return df

def bdp(tickers,fld,orides={},valtype='float'):
    #tickers must be a list
    #fld is a String with bloomberg field (only 1 fld at a time is accepted)
    #orides is a dictionary of override fields. {'override','value}. (OPTIONAL)
    #valtype is the type of value expected , (OPTIONAL, defaults to float)
    
    sessionOptions = blpapi.SessionOptions()
    session = blpapi.Session(sessionOptions)
    session.start()
    session.openService("//blp/refdata")
    refDataService = session.getService("//blp/refdata")
    request = refDataService.createRequest("ReferenceDataRequest")
    for t in tickers:
        request.getElement('securities').appendValue(t)
    request.getElement("fields").appendValue(fld)
    if bool(orides):
        for k,v in orides.items():
            h = request.getElement('overrides').appendElement()
            h.setElement('fieldId',k)
            h.setElement('value',v)
    
    session.sendRequest(request)
    dct={}
    while(True):
        ev = session.nextEvent(500)
        for msg in ev:
            if msg.messageType() == 'ReferenceDataResponse':
                for v in msg.getElement('securityData').values():
                    sec = v.getElementAsString('security')
                    if valtype=='str':
                        try:
                            val = v.getElement('fieldData').getElementAsString(fld)
                        except:val=np.nan
                    elif valtype=='float':
                        try:
                            val = v.getElement('fieldData').getElementAsFloat(fld)
                        except: val=np.nan
                        
                    dct[sec]=val
        if ev.eventType() == blpapi.Event.RESPONSE:
                break
    session.stop()
    return dct

