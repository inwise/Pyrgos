# -*- coding: utf-8 -*-

__author__ = 'Alexander Grechko'

from math import sqrt,log,exp
from scipy.stats import norm
from scipy import optimize,stats
from numpy import *
import datetime

def get_t(mat_date,date=None):
    #возвращает кол-во дней до даты исполнения
    today=date
    if today==None:
        today=datetime.datetime.today()
    t=(mat_date-today).total_seconds()/60/60/24/365
    return t

def euro_call_option_value(s,k):
    #функция выплаты европейского опциона call
    return max(s-k,0)

def euro_put_option_value(s,k):
    #функция выплаты европейского опциона put
    return max(k-s,0)

def euro_option_call_monte_carlo(s,t,t1,v,r,k):
    #вычисляет стоимость европейского опциона call по методу Монте-Карло
    n=10000000
    dt=t1-t
    z=norm.rvs(size=n)
    sum=0.0
    for i in range(n):
        sum+=euro_call_option_value(s*exp((r-v*v/2.0)*dt+v*sqrt(dt)*z[i]),k)
    return sum/n

def euro_option_gamma(s,t,t1,v,r,k):
    #вычисляет гамму европейского опциона
    d1=get_d1(s,t,t1,v,r,k)
    return norm.pdf(d1)/(s*v*sqrt(t1-t))

def get_d1(s,t,t1,v,r,k):
    return (log(s/k)+(r+v*v/2.0)*(t1-t))/(v*sqrt(t1-t))

def euro_option_call_pricing(s,t,t1,v,r,k):
    #возвращает справедливую стоимость европейского опциона call по формуле Блека-Шоулза
    #s - price of stock, v - volatility, t - current time, t1 - last time, r - stavka, k - strike
    d1=get_d1(s,t,t1,v,r,k)
    d2=d1-v*sqrt(t1-t)
    u=s*norm.cdf(d1)-k*exp(-r*(t1-t))*norm.cdf(d2)
    return u

def euro_option_put_pricing(s,t,t1,v,r,k):
    #возвращает справедливую стоимость европейского опциона put по формуле Блека-Шоулза
    #s - price of stock, v - volatility, t - current time, t1 - last time, r - stavka, k - strike
    return euro_option_call_pricing(s,t,t1,v,r,k)-s+k*exp(-r*(t1-t))

def option_volatility(s,t,t1,r,k,o,pricing_func):
    #считает волатильность базового актива по текущей цене опциона, pricing_func - произвольная функция цены

    def F(v):
        return pricing_func(s,t,t1,v,r,k)-o

    #x=optimize.root(F,[0.5,]).x
    min_vol=0.00001
    if F(min_vol)>0:
        return 0
    max_vol=1.0
    while (F(max_vol)<0):
        max_vol*=2
        if max_vol>100000:
            return -1
    x=optimize.bisect(F,min_vol,max_vol)
    return x


def euro_option_call_volatility(s,t,t1,r,k,o):
    #возвращает прогнозную волатильность акции s при условии цены со страйком k и временем исполнения t1
    #для европейского опциона call
    return option_volatility(s,t,t1,r,k,o,euro_option_call_pricing)

def euro_option_put_volatility(s,t,t1,r,k,o):
    #возвращает прогнозную волатильность акции s при условии цены со страйком k и временем исполнения t1
    #для европейского опциона put
    return option_volatility(s,t,t1,r,k,o,euro_option_put_pricing)

def american_option_put_volatility(s,t,t1,r,k,o):
    #возвращает прогнозную волатильность акции s при условии цены со страйком k и временем исполнения t1
    #для американского опциона put
    return option_volatility(s,t,t1,r,k,o,american_option_put_pricing_binomial_tree)

def option_pricing_binomial_tree(s,t,t1,v,r,k,pay_func):
    #считает текущую стоимость обязательства с функцией выплаты в конце периода pay_func биномиальная модель
    n=1000
    dt=(t1-t)/n
    rn=exp(r*dt)-1
    u=exp(v*sqrt(dt))
    d=1/u
    pu=(1+rn-d)/(u-d)
    pd=1-pu
    f=[pay_func(s*(u**i)*(d**(n-i)),k) for i in range(n+1)]
    for k in range(n,0,-1):
        fp=[1/(1+rn)*(f[i+1]*pu+f[i]*pd) for i in range(k)]
        f=fp
    return f[0]

def euro_option_call_pricing_binomial_tree(s,t,t1,v,r,k):
    #считает европейский call по биномиальной модели
    return option_pricing_binomial_tree(s,t,t1,v,r,k,euro_call_option_value)

def euro_option_put_pricing_binomial_tree(s,t,t1,v,r,k):
    #считает европейский put по биномиальной модели
    return option_pricing_binomial_tree(s,t,t1,v,r,k,euro_put_option_value)

def american_option_pricing_binomial_tree(s,t,t1,v,r,k,pay_func):
    #вычисляет цену американского опциона с функцией выплаты pay_func по биномиальной модели
    n=1000
    dt=(t1-t)/n
    rn=exp(r*dt)-1
    u=exp(v*sqrt(dt))
    d=1/u
    pu=(1+rn-d)/(u-d)
    pd=1-pu
    f=[pay_func(s*(u**i)*(d**(n-i)),k) for i in range(n+1)]
    for j in range(n,0,-1):
        fp=[max(1/(1+rn)*(f[i+1]*pu+f[i]*pd),pay_func(s*(u**i)*(d**(j-1-i)),k)) for i in range(j)]
        f=fp
    return f[0]

def american_option_call_pricing_binomial_tree(s,t,t1,v,r,k):
    #возвращает цену американского опциона call по биномиальной модели
    return american_option_pricing_binomial_tree(s,t,t1,v,r,k,euro_call_option_value)

def american_option_put_pricing_binomial_tree(s,t,t1,v,r,k):
    #возвращает цену американского опциона put по биномиальной модели
    return american_option_pricing_binomial_tree(s,t,t1,v,r,k,euro_put_option_value)
