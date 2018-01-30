# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

import shelve
from datetime import datetime,timedelta
import rtsvix
from scipy.integrate import simps
import numpy as np
import option
from quotes import load_future_quotes_from_hdf_store,valid_time
import pandas as pd
import matplotlib.pyplot as plt

def to_datetime(date):
    #преобразует ключевую строку в дату-время
    return datetime.strptime(date,"%d.%m.%Y %H:%M:%S")

def levy_expected_quadratic_variation(opt_dict,t):
    valid_options = rtsvix.get_valid_options(opt_dict)
    to_array = lambda options,i: [option[i] for option in options]
    k0,f,call_options,put_options = valid_options[0],valid_options[1],valid_options[3],valid_options[2]
    k_call = np.array(to_array(call_options,0))
    price_call = np.array(to_array(call_options,1))
    k_put = np.array(to_array(put_options,0))
    price_put = np.array(to_array(put_options,1))
    log_k0 = np.log(k0)
    #print 'Put strikes: ',k_put
    #print 'Call strikes: ',k_call
    #print 'K0: ',k0
    #print 'F: ',f
    expected = log_k0+f/k0-1-simps(price_call/k_call**2,k_call)-simps(price_put/k_put**2,k_put)\
               -price_put[0]/k_put[0]**2*5000.0-price_call[0]/k_call[0]**2*(k_call[0]-k_put[-1])
    #print expected
    expected_quad = log_k0**2+2*log_k0*(f/k0-1)+2*(simps(price_call*(1-np.log(k_call))/k_call**2,k_call)
                                                   +simps(price_put*(1-np.log(k_put))/k_put**2,k_put)
                                                   +price_put[0]*(1-np.log(k_put[0]))/k_put[0]**2*5000.0
                                                   +price_call[0]*(1-np.log(k_call[0]))/k_call[0]**2*(k_call[0]-k_put[-1]))
    #print expected_quad
    var=expected_quad-expected**2
    #print var
    #print "VIX: ",np.sqrt(-2.0*(expected-np.log(f))/t)*100.0
    return var

def levy_expected_quadratic_variation2(opt_dict,t):
    valid_options = rtsvix.get_valid_options(opt_dict)
    to_array = lambda options,i: [option[i] for option in options]
    k0,f,call_options,put_options = valid_options[0],valid_options[1],valid_options[3],valid_options[2]
    k_call = np.array(to_array(call_options,0))
    price_call = np.array(to_array(call_options,1))
    k_put = np.array(to_array(put_options,0))
    price_put = np.array(to_array(put_options,1))
    log_k0 = np.log(k0)
    #print 'Put strikes: ',k_put
    #print 'Call strikes: ',k_call
    #print 'K0: ',k0
    #print 'F: ',f
    sum1 = price_call[0]/k_call[0]**2*(k_call[0]-k_put[-1])
    for i in range(1,len(k_call)):
        sum1 += price_call[i]/k_call[i]**2*(k_call[i]-k_call[i-1])
    sum2 = price_put[0]/k_put[0]**2*5000.0
    for i in range(1,len(k_put)):
        sum2 += price_put[i]/k_put[i]**2*(k_put[i]-k_put[i-1])
    expected = log_k0+f/k0-1-sum1-sum2
    #print expected
    sum1 = price_call[0]*(1-np.log(k_call[0]))/k_call[0]**2*(k_call[0]-k_put[-1])
    for i in range(1,len(k_call)):
        sum1 += price_call[i]*(1-np.log(k_call[i]))/k_call[i]**2*(k_call[i]-k_call[i-1])
    sum2 = price_put[0]*(1-np.log(k_put[0]))/k_put[0]**2*5000.0
    for i in range(1,len(k_put)):
        sum2 += price_put[i]*(1-np.log(k_put[i]))/k_put[il]**2*(k_put[i]-k_put[i-1])
    expected_quad = log_k0**2+2*log_k0*(f/k0-1)+2*(sum1+sum2)
    #print expected_quad
    var=expected_quad-expected**2
    #print var
    #print "VIX: ",np.sqrt(-2.0*(expected-np.log(f))/t)*100.0
    return var,expected_quad,var/(-expected+np.log(f))

'''
def vol_index_single(option_dict,t):
    #индекс по ближайщей серии
    mat_dates = option_dict.keys()
    mat_dates.sort()
    t = option.get_t(mat_dates[0],to_datetime())
    return 1/t*levy_expected_quadratic_variation()
'''

def vol_index_(date,option_dict):
    mat_dates = option_dict.keys()
    mat_dates.sort()
    t1 = option.get_t(mat_dates[0],to_datetime(date))
    t2 = option.get_t(mat_dates[2],to_datetime(date))
    var,exp_quad,qx=levy_expected_quadratic_variation2(option_dict[mat_dates[0]],t1)
    v1 = 1/t1*var
    #print "V1: ",np.sqrt(v1)*100.0
    v2 = 1/t2*levy_expected_quadratic_variation2(option_dict[mat_dates[2]],t2)[0]
    #print "V2: ",np.sqrt(v2)*100.0
    t30 = 30.0/365.0
    return 100.0*np.sqrt(v1),100.0*np.sqrt(1.0/t30*np.abs(t1*v1*(t2-t30)/(t2-t1)+t2*v2*(t30-t1)/(t2-t1))),\
           100.0*np.sqrt(exp_quad/t1),qx

last_value = None

def future_rv_(date,store,td = timedelta(hours=1)):
    global last_value
    trim_seconds = lambda date: datetime(date.year,date.month,date.day,date.hour,date.minute)
    trim_dt = trim_seconds(date)
    if last_value<>None:
        if trim_dt==last_value[0]:
            return last_value[1]
    df=load_future_quotes_from_hdf_store(store,'RTSI',date,date+td)
    #print df
    #print len(df)
    log = np.log(df['close'])
    rv = np.sqrt(((log-log.shift(1))**2).sum()*365*24)*100
    last_value = (trim_dt,rv)
    return rv

def future_rv2_(date,store):
    store = pd.HDFStore('../moex.h5')
    df=load_future_quotes_from_hdf_store(store,'RTSI',date,date+timedelta(hours=1))
    log = np.log(df['close'])-np.log(df['close'].shift(1))
    return np.sqrt(((log-log.shift(1))**2).sum()*365*14)*100

to_float = lambda s: np.float(s.replace(',','.'))

def vol_indexes_(date,d):
    v1,v2,exp_quad,qx = vol_index_(date,d[date]['option_dict'])
    return v1,v2,to_float(d[date]['RVI']),future_rv(to_datetime(date)),exp_quad,qx

def search_start_i(dates,start_date):
    i=0
    dt=to_datetime(dates[i])
    while not (dt>start_date and valid_time(dt)):
        dt=to_datetime(dates[i])
        i+=1
    return i

store = pd.HDFStore('../moex.h5')
future_rv = lambda date: future_rv_(date,store)
future_rv2 = lambda date: future_rv2_(date,store)
d = shelve.open('../options','r')
dates = d.keys()
dates.sort(key = lambda x: to_datetime(x))
vol_index = lambda date: vol_index_(date,d[date]['option_dict'])
#i = 3715
i = search_start_i(dates,to_datetime('23.09.2015 10:05:00'))
l = 1600 # кол-во данных
print dates[i],' ',dates[i+l]
#print vol_index(dates[i])
'''
mat_dates = option_dict.keys()
mat_dates.sort()
t = option.get_t(mat_dates[0],to_datetime(dates[i]))
print np.sqrt(levy_expected_quadratic_variation(option_dict[mat_dates[0]],t)/t)*100.0
'''
'''
print 'RVI: ',d[dates[i]]['RVI']
print 'RTSVX: ',d[dates[i]]['RTSVX']
print 'NRTSVX: ',d[dates[i]]['NRTSVX']
print 'Ri: ',d[dates[i]]['Ri']
print future_rv(to_datetime(dates[i]))
'''
vol_indexes = lambda date: vol_indexes_(date,d)
vol_indexes_v = np.vectorize(vol_indexes)
vols=vol_indexes_v(dates[i:i+l])
store.close()
t=np.arange(len(vols[0]))

#plt.plot(t,vols[0],label = 'Quad. var.')

#plt.plot(t,vols[1],label = 'quad')
#plt.plot(t,vols[2],label = 'RVI')
#plt.plot(t,vols[3],label = 'RV')

#plt.plot(t,vols[4],label = 'E[ln^2(S_t)]')
plt.plot(t,vols[5],label = '$Q_x$')
#rv_mean = vols[3].mean()
#plt.axhline(y=rv_mean,label = 'RV average')
plt.legend()
plt.show()