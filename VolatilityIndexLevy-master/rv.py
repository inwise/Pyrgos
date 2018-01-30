# -*- coding: utf-8 -*-

import datetime
from math import sqrt
import win32com.client
import quotes

__author__ = 'Александр Гречко'

base_p_code='RTSI'

ad = win32com.client.Dispatch("ADLite.AlfaDirect")
if not ad.Connected:
    ad.UserName="agrechko"
    ad.Password="s8x1g5"
    ad.Connected=True
#tomorrow=(datetime.datetime.today()-datetime.timedelta(days=1)).date()
date_from=datetime.datetime.strptime('09.01.2015 10:00',"%d.%m.%Y %H:%M")
date_to=datetime.datetime.strptime('09.01.2015 11:00',"%d.%m.%Y %H:%M")
df=quotes.load_quotes_from_ad_df(ad,'INDEX',base_p_code,0,date_from,date_to)
print df
'''
rates=quotes.get_rates_of_return(candles)
sum=0
n=len(rates)
for i in range(n):
    #sum+=(rates[i+1]-rates[i])**2
    sum+=rates[i]**2
print sqrt(365*sum)
'''