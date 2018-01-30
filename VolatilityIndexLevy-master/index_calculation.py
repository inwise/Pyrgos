# -*- coding: utf-8 -*-

import win32com.client
from numpy import *
import matplotlib.pyplot as plt
import rtsvix
import quotes
import datetime

__author__ = 'Александр Гречко'
ad=None
PlaceCode="INDEX"

def load_nrtsvx():
    d=list()
    v=list()
    with open('rtsvix.csv','r') as f:
        for line in f:
            a=line.split(';')
            d.append(datetime.datetime.strptime(a[0],"%d.%m.%Y %H:%M:%S"))
            v.append(float(a[1]))
    return (d,v,)

def nrtsvx_period(data,date_from,date_to):
    d,v=data
    i1=0
    for i in range(len(d)):
        if d[i]>=date_from:
            i1=i
            break
    i2=i1
    for i in range(i1,len(d)):
        if d[i]>date_to:
            i2=i
            break
    return (d[i1:i2],v[i1:i2])

def nrtsvx_day(data):
    d,v=data
    nd=list()
    nv=list()
    date=d[0].date()
    for i in range(len(d)):
        ndate=d[i].date()
        if ndate!=date:
            nd.append(date)
            nv.append(v[i])
        date=ndate
    return (nd,nv)

def nrtsvx_align(data,rtsvx_t):
    d,v=data
    nd=list()
    nv=list()
    j=0
    for i in range(len(rtsvx_t)):
        while j<len(d) and d[j]<rtsvx_t[i]:
            j+=1
        if j==len(d):
            break
        nd.append(d[j])
        nv.append(v[j])
    return (nd,nv)

def convert_candles(candles):
    n=len(candles)
    print n
    t=list()
    p=empty(n)
    for i in range(n):
        t.append(candles[i].date)
        p[i]=candles[i].close
    return (t,p,)

def plot_nrtsvx():
    data=load_nrtsvx()
    #date_from=datetime.datetime.strptime('03.06.2013 10:00',"%d.%m.%Y %H:%M")
    #date_to=datetime.datetime.strptime('07.06.2013 23:50',"%d.%m.%Y %H:%M")
    date_from=datetime.datetime.strptime('03.06.2013',"%d.%m.%Y")
    date_to=datetime.datetime.strptime('07.06.2013',"%d.%m.%Y")
    nrtsvx_d,nrtsvx_v=nrtsvx_period(data,date_from,date_to)
    candles=quotes.load_quotes_from_ad_server(ad,PlaceCode,'RTSVX',4,date_from,date_to)
    t,p=convert_candles(candles)
    plt.plot(t,p,label="RTSVX")
    plt.legend()
    nrtsvx_d,nrtsvx_v=nrtsvx_align((nrtsvx_d,nrtsvx_v),t)
    plt.plot(nrtsvx_d,nrtsvx_v,label="NRTSVX")
    plt.legend()
    plt.show()

def report():
    data=load_nrtsvx()
    date_from=datetime.datetime.strptime('03.06.2013',"%d.%m.%Y")
    date_to=datetime.datetime.strptime('07.06.2013',"%d.%m.%Y")
    candles=quotes.load_quotes_from_ad_server(ad,PlaceCode,'RTSVX',5,date_from,date_to)
    t,p=convert_candles(candles)
    nrtsvx_d,nrtsvx_v=nrtsvx_align(data,t)
    with open('report.csv','w') as f:
        f.write("Дата и время;RTSVX;NRTSVX;Абс. погрешность;Отн. погрешность %\n")
        for i in range(len(t)):
            if abs(t[i]-nrtsvx_d[i])<datetime.timedelta(hours=1):
                f.write("%s;%g;%.2f;%g;%.2f\n"%(t[i].strftime("%d.%m.%Y %H:%M"),p[i],nrtsvx_v[i],nrtsvx_v[i]-p[i],
                                         (nrtsvx_v[i]-p[i])/nrtsvx_v[i]*100))

def rv(day,tf=0):
    date_from=datetime.datetime.strptime(day+' 10:00',"%d.%m.%Y %H:%M")
    date_to=datetime.datetime.strptime(day+' 23:55',"%d.%m.%Y %H:%M")
    candles=quotes.load_quotes_from_ad_server(ad,'INDEX','RTSI',tf,date_from,date_to)
    if candles is None:
        print day
    candles=quotes.fill_holes(candles)
    rates=quotes.get_rates_of_return(candles)
    sum=0
    n=len(rates)
    for i in range(n-1):
        sum+=(rates[i+1]-rates[i])**2
    return sqrt(365*sum)*100

def report_rv():
    days=('28.05.2013','29.05.2013','30.05.2013','31.05.2013','03.06.2013','04.06.2013','05.06.2013','06.06.2013',
          '07.06.2013','10.06.2013','11.06.2013')
    dts=list()
    for day in days:
        dts.append(datetime.datetime.strptime(day,"%d.%m.%Y"))
    t,v=load_nrtsvx()
    nv=list()
    i=0
    for dt in dts:
        for j in range(i,len(t)):
            if dt.date()==t[j].date():
               break
        i=j
        nv.append(v[i])
    print nv
    rvs=list()
    rv5s=list()
    for day in days:
        rvs.append(rv(day))
        rv5s.append(rv(day,1))
    print rvs
    print rv5s
    print len(rvs)
    with open('rv.csv','w') as f:
        f.write("Дата;NRTSVX;RV;RV5min;Абс. ошибка RV;Отн. ошибка RV %;Абс. ошибка Rv5min;Отн. ошибка RV5min %\n")
        for i in range(len(days)):
            f.write("%s;%.2f;%.2f;%.2f;%.2f;%.2f;%.2f;%.2f\n"%(days[i],nv[i],rvs[i],rv5s[i],rvs[i]-nv[i],
                                                               (rvs[i]-nv[i])/rvs[i]*100,rv5s[i]-nv[i],
                                                               (rv5s[i]-nv[i])/rv5s[i]*100))

ad = win32com.client.Dispatch("ADLite.AlfaDirect")
if not ad.Connected:
    ad.UserName="agrechko"
    ad.Password="s8x1g5"
    ad.Connected=True
option_dict=rtsvix.get_option_dict(ad)
rtsi=rtsvix.get_current_rtsi(ad)
print rtsi
#rtsvix.plot_implied_volatility_smile(option_dict,rtsi[0])
rtsvix.plot_rtsvix()
#plot_nrtsvx()
#report()
#date_from=datetime.datetime.strptime('04.06.2013 18:40',"%d.%m.%Y %H:%M")
#date_to=datetime.datetime.strptime('04.06.2013 19:05',"%d.%m.%Y %H:%M")
#candles=quotes.load_quotes_from_ad_server(ad,PlaceCode,'RTSVX',0,date_from,date_to)
#print candles[6].close
#report_rv()
