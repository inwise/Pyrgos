# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

import datetime
import win32com.client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rtsvix import rtsvix,get_option_dict,get_current_rtsi,get_last_rtsvx_info,plot_implied_volatility_smile,test_arbitrage_oppportunities
from adengine import adengine
from strategy import order

account="55479-000"
PlaceCode="FORTS"

ad = win32com.client.Dispatch("ADLite.AlfaDirect")
eng = adengine(ad)
fig = plt.figure()
smile_plot = fig.add_subplot(1,1,1)
sended = True

def valid_time():
    #проверяет условия времени, торговое время и не клиринг
    now=datetime.datetime.now()
    if now.hour<10:
        return False
    if now.hour==23 and now.minute>=45:
        return False
    if now.hour==18 and now.minute>=45:
        return False
    if now.hour==14 and now.minute<=3:
        return False
    return True

def get_option_code(opt_data,call=True):
    #возвращает код опциона opt_data
    call_codes=('A','B','C','D','E','F','G','H','I','J','K','L',)
    put_codes=('M','N','O','P','Q','R','S','T','U','V','W','X',)
    m=''
    if call:
        m=call_codes[opt_data.mat_date.month-1]
    else:
        m=put_codes[opt_data.mat_date.month-1]
    return "RI%dB%s4"%(int(opt_data.strike),m,)

def monitor(i):
    if not ad.Connected:
        ad.UserName = "agrechko"
        ad.Password = "s8x1g5"
        ad.Connected = True
    try:
        res = rtsvix(ad)
        if res!=None:
            value,date = res
        option_dict = get_option_dict(ad)
        rtsi,d,rtsi_sell,rtsi_buy = get_current_rtsi(ad)
        rtsvx = get_last_rtsvx_info(ad)[0]
    except TypeError:
        return
    test_arbitrage_oppportunities(option_dict,rtsi,rtsi_sell,rtsi_buy)
    smile_plot.clear()
    smile_plot.set_xlabel("K")
    smile_plot.set_ylabel(r'$\sigma$')
    smile_plot.set_title('volatility smile')
    plot_implied_volatility_smile(option_dict,rtsi,smile_plot)
    if res!=None:
        smile_plot.axhline(y=value/100.0,color='r',label='NRTSVX')
    smile_plot.axhline(y=float(rtsvx.replace(',','.'))/100.0,color='m',label='RTSVX')
    smile_plot.axvline(x=rtsi,color='g',label='RTSI')
    smile_plot.legend()


ani=animation.FuncAnimation(fig,monitor,interval = 15000)
plt.show()
