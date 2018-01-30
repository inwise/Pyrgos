# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.integrate import quad,romberg,simps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

import matplotlib
matplotlib.style.use('ggplot')

'''
import sys

reload(sys)
sys.setdefaultencoding('cp1251')
'''
__author__ = 'Александр Гречко'

def load_quotes_from_finam_file_df(fn):
    #загружает котировки из файла finam в формате pandas.DataFrame
    f = open(fn)
    names = ('date','time','open', 'high', 'low', 'close', 'volume',)
    types = {'open': np.float64, 'high': np.float64, 'low': np.float64, 'close': np.float64, 'volume': np.int64}
    df = pd.read_csv(f, parse_dates={'datetime': [0,1]}, skiprows=1, names=names, dtype=types, index_col=0)
    f.close()
    return df

def rates_of_return(x):
    #возвращает доходности, x - логарифм цен
    return (x-x.shift(1))[1:]

def log_price(df):
    #возвращает логарифм доходностей цен закрытия
    return np.log(df.close)

def power_variation(x,p,c=10.0):
    #вычисляет power variation логдоходностей x и степени p
    #if hasattr(p,"__len__"):
    #    return np.array([power_variation(x,pi,c) for pi in p])
    if p >=2 :
        return (np.abs(x) ** p).sum()
    xa = np.abs(x)
    i = xa <= c
    #print (i == False).sum()
    xp = xa[i] ** p
    return xp.sum()

def asf(x,p,c=10.0):
    #activity signature function
    r = rates_of_return(x)
    x2 = x.resample("2min", how="last", closed='right', label='right')
    r2 = rates_of_return(x2)
    log_pv2 = np.log(power_variation(r2,p,c))
    log_pv1 = np.log(power_variation(r,p,c))
    b = np.log(2)*p/(np.log(2)+log_pv2-log_pv1)
    #if np.isnan(b):
    #    print "Fail p=",p
    return b

p = np.linspace(1.0,2.0,num=3)
#p = np.linspace(0.1,4.0,num=100)

def asfp(x,c=10.0):
    #asf для различных p из интервала [0,4]
    #return [asf(x,p0) for p0 in p]
    return pd.DataFrame([[asf(x,p0,c) for p0 in p]], columns=p)
    #return dict([(p0,asf(x,p0,c)) for p0 in p])

c=10.0

def clear_data(x):
    #очистка и подготовка данных
    date = x.index[0].date()
    xd = x[:str(date)+" 18:40"].asfreq(pd.offsets.Minute(), method="pad")
    return xd

def asfp_raw(x):
    #тоже что asfp только еще выполняет подготовку и очистку данных
    xd = clear_data(x)
    return asfp(xd,c)

tau = 0.1

def asfp_raw_point_estimator(x):
    #очистка и считает интегралы по asf за каждый день
    xd = clear_data(x)
    btau = asf(xd,tau,c)
    if btau<0:
        print "Fail"
    #if np.isnan(btau):
    #    print "Fail"

    v,e = quad(lambda p: asf(xd,p,c),tau,btau)
    if np.isnan(v):
        print "Fail"

    #p = np.linspace(tau,btau,num=100)
    #y = [asf(xd,pi,c) for pi in p]
    #if np.isnan(np.sum(p)):
    #    print "Fail"
    #v = np.trapz(y,p)
    #v = simps(y,p)
    #v = romberg(lambda p: asf(xd,p,c),tau,btau)
    r = 1.0/(btau-tau)*v
    return r

def mean_confidence_interval(data, confidence=0.95):
    n = len(data.index)
    print "n=", n
    m = data.mean()
    se = data.sem()
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m,m-h,m+h

'''
fn = "RTSI_140915_150915.csv"
df = load_quotes_from_finam_file_df(fn)
x=log_price(df)
print 'max of x = ',x.values.max()
grouped = x.groupby(x.index.date)


res = grouped.apply(asfp_raw_point_estimator)
print res
print type(res)
res.hist(bins=100)
m,a,b = mean_confidence_interval(res)
print "mean = ",m," confidence interval (%g,%g)"%(a,b)
plt.show()
'''

'''
res = grouped.apply(asfp_raw)
res.index = [index[0] for index in res.index]
print res
print type(res)
print 'max of res = ',res.values.max()
print 'minx of res = ',res.values.min()
'''
'''
for s in p:
    res[s].plot(label=str(s))
plt.legend()
plt.title("ASF c="+str(c))
plt.show()
'''

'''
res.quantile(0.25).plot(label="0.25")
res.quantile(0.5).plot(label="0.5")
res.quantile(0.75).plot(label="0.75")
'''
'''
qs=np.arange(0.1,1.0,0.1)
for q in qs:
    res.quantile(q).plot(label=str(q))
plt.legend()
plt.title("QASF c="+str(c))
plt.show()
'''
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print len(res.index)
P, T = np.meshgrid(res.columns.values,np.arange(len(res.index)))
print 'T=',T
print T.shape
print 'P=',P
print P.shape
Z = res.as_matrix()
print 'Z=',Z
print Z.shape
ax.plot_surface(P,T,Z)
#print 'min=',np.nanmin(Z)
#print 'max=',np.nanmax(Z)
ax.set_zlim(-3.0,3.0)
plt.show()
'''
