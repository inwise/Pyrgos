# -*- coding: utf-8 -*-

import win32com.client
from QuantLib import *
import rtsvix

__author__ = 'Александр Гречко'

def datetime_to_date(dt):
    #преобразует datetime в Date
    return Date(dt.day,dt.month,dt.year)

ad = win32com.client.Dispatch("ADLite.AlfaDirect")
if not ad.Connected:
    ad.UserName="agrechko"
    ad.Password="s8x1g5"
    ad.Connected=True

mat_dt,valid_options=rtsvix.valid_options_from_ad(ad)
mat_date=datetime_to_date(mat_dt)
risk_free_rate=FlatForward(mat_date,0,Actual365Fixed())
dividend_yield=FlatForward(mat_date,0,Actual365Fixed())

helpers=list()
put_options=valid_options[2]
call_options=valid_options[3]
for put_option in put_options:
    HestonModelHelper()
    #helpers.append(HestonModelHelper())

ival={'v0': 0.1, 'kappa': 1.0, 'theta': 0.1, 'sigma': 0.5, 'rho': -.5}
spot=SimpleQuote(1290.0)
process=HestonProcess(YieldTermStructureHandle(risk_free_rate),YieldTermStructureHandle(dividend_yield),
                      QuoteHandle(spot),ival['v0'],ival['kappa'],ival['theta'],ival['sigma'],ival['rho'])
model=HestonModel(process)
engine=AnalyticHestonEngine(model)
calendar=Russia()
print "OK!"
