# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

import shelve
from datetime import datetime
import option

def to_datetime(date):
    #преобразует ключевую строку в дату-время
    return datetime.strptime(date,"%d.%m.%Y %H:%M:%S")

def get_valid_options(opt_dict,s,t):
    #возвращает список ликвидных опционов с признаком call или put
    opts=list()
    strikes=opt_dict.keys()
    c=1.2
    calls=0
    for strike in strikes:
        opt=opt_dict[strike]
        if opt.call_buy!=0 and opt.call_sell!=0:
            if opt.call_sell/opt.call_buy<c:
                midprice=(opt.call_buy+opt.call_sell)/2
                vol=option.euro_option_call_volatility(s,0,t,0,strike,midprice)
                opts.append((strike,midprice,True,vol,))
                calls+=1
        if opt.put_buy!=0 and opt.put_sell!=0:
            if opt.put_sell/opt.put_buy<c:
                midprice=(opt.put_buy+opt.put_sell)/2
                vol=option.euro_option_put_volatility(s,0,t,0,strike,midprice)
                opts.append((strike,midprice,False,vol,))
    print "calls: ",calls
    return opts

d=shelve.open('c:\OneDrive\options_09_2014','r')

dates=d.keys()
dates.sort(key=lambda x: to_datetime(x))
#дата и время это ключ словаря d
print dates[2]
print dates[-1150]

#первая запись
date1=dates[-1150]
data=d[date1]
print data['Ri']
option_dict=data['option_dict']
mat_dates=option_dict.keys()
mat_dates.sort()
print mat_dates
print len(option_dict[mat_dates[1]].keys())
t=option.get_t(mat_dates[2],to_datetime(date1))
print 't=',t
opts1=get_valid_options(option_dict[mat_dates[2]],data['Ri'],t)
print len(opts1)
print opts1
