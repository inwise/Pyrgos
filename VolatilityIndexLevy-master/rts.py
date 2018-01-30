# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

import redis
import cPickle as pickle
import pandas as pd
import datetime

r=redis.Redis()

def save_rts_data(datestr):
    date = datetime.datetime.strptime(datestr, "%d.%m.%Y").date()
    values = r.lrange('rts:' + datestr + ':values', 0, -1)
    values = [pickle.loads(value) for value in values]
    ind = [datetime.datetime.combine(date, datetime.datetime.strptime(value[0], "%H:%M:%S").time()) for value in values]
    values = [value[1] for value in values]
    rts = pd.Series(values, index=ind)
    store = pd.HDFStore("../../trades/rts.h5")
    store.append('RTS', rts)
    store.close()

def clear_rts_data(datestr):
    pipe = r.pipeline()
    path = 'rts:'+datestr+':'
    pipe.delete(path+'last-i')
    pipe.delete(path+'values')
    pipe.execute()

datestr='28.07.2017'
save_rts_data(datestr)
clear_rts_data(datestr)
