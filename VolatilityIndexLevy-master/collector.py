# -*- coding: utf-8 -*-

import os
import datetime
import win32com.client
from twisted.internet import task
from twisted.internet import reactor
import shelve
from rtsvix import rtsvix,get_option_dict,get_current_rtsi,get_last_rtsvx_info,get_last_rvi_info

__author__ = 'Александр Гречко'

ad = win32com.client.Dispatch("ADLite.AlfaDirect")

def collect():
    if not ad.Connected:
        ad.UserName="agrechko"
        ad.Password="s8x1g5"
        ad.Connected=True
    last_collect=None
    if os.path.exists('../../last_collect_09_2015'):
        with open('../../last_collect_09_2015') as f:
            str=f.readline()
            last_collect=datetime.datetime.strptime(str,"%d.%m.%Y %H:%M:%S\n")
    try:
        res=rtsvix(ad)
        if res==None:
            value=0
            date=datetime.datetime.today()
        else:
            value,date=res
        option_dict=get_option_dict(ad)
        rtsi=get_current_rtsi(ad)[0]
        rtsvx=get_last_rtsvx_info(ad)[0]
        rvi=get_last_rvi_info(ad)[0]
    except TypeError:
        return
    if (last_collect==None) or (date>last_collect):
        date_string=date.strftime("%d.%m.%Y %H:%M:%S")
        with open('../../rtsvix_09_2015.csv','a') as f:
            f.write("%s;%g\n"%(date_string,value))
        with open('../../last_collect_09_2015','w') as f:
            f.write(date_string+"\n")
        d=shelve.open('../../options_09_2015')
        d[date_string]={'option_dict': option_dict,'Ri': rtsi,'RTSVX': rtsvx,'NRTSVX': value,'RVI': rvi}
        d.close()

l=task.LoopingCall(collect)
l.start(15)
reactor.run()
