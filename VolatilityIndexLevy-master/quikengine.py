# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

from adengine import engine
import redis
import cPickle as pickle
import datetime
import pandas as pd

class quikengine(engine):
    #интерфейс для торговли через терминал Quik

    def __init__(self):
        self.r=redis.Redis()
        self.indexes=dict()
        self.mins=dict()

    def trades(self,date,pcode,first=0,last=-1):
        #возвращает список сделок из хранилища Redis в формате pandas.DataFrame
        lname='trades:'+date.strftime("%d.%m.%Y")+':'+pcode
        trades=self.r.lrange(lname,first,last)
        trades=[pickle.loads(trade) for trade in trades]
        ind=[datetime.datetime.combine(date,datetime.datetime.strptime(trade[2],"%H:%M:%S").time()) for trade in trades]
        trades=[(int(trade[1]),trade[4],trade[5],True if trade[7]==u'Купля' else False) for trade in trades]
        trades_df=pd.DataFrame(trades,index=ind,columns=('id','price','volume','buy'))
        return trades_df

    def clear_all_date_trades(self,date):
        #очищает все данные по торгам по всем инструментам за день date
        pipe = self.r.pipeline()
        pcodes = self.all_date_pcodes(date)
        date_str = date.strftime("%d.%m.%Y")
        for pcode in pcodes:
            lname = 'trades:' + date_str + ':' + pcode
            pipe.delete(lname)
        pipe.delete('pcodes:'+date_str)
        pipe.srem('dates',date_str)
        pipe.execute()

    def clear_all_trades(self):
        #очищает данные по всем инструментам по всем хранимым датам
        dates = self.r.smembers('dates')
        today = datetime.date.today()
        for date_str in dates:
            date = datetime.datetime.strptime(date_str, "%d.%m.%Y").date()
            if date!=today:
                self.clear_all_date_trades(date)

    def all_date_pcodes(self,date):
        #возвращает список сохраненных инструментов за дату date
        date_str = date.strftime("%d.%m.%Y")
        pcodes = self.r.smembers('pcodes:' + date_str)
        return pcodes

    def save_all_date_trades(self,date,store_path):
        #сохраняет все сделки за день date_str в формате dd.mm.YY файл store_path
        store = pd.HDFStore(store_path+'/'+date.strftime("%Y-%m-%d")+'.h5')
        pcodes = self.all_date_pcodes(date)
        for pcode in pcodes:
            trades=self.trades(date,pcode)
            key='FORTS/'+pcode
            store.append(key,trades)
        store.close()

    def save_all_trades(self,store_path):
        #сохраняет все сделки за день date_str в формате dd.mm.YY файл store_path
        dates=self.r.smembers('dates')
        today=datetime.date.today()
        for date_str in dates:
            date=datetime.datetime.strptime(date_str,"%d.%m.%Y").date()
            if date!=today:
                self.save_all_date_trades(date,store_path)

    def load_quotes(self,pcode,period, from_time, to_time):
        if not self.indexes.has_key(pcode):
            self.indexes[pcode]=0
        index=self.indexes[pcode]
        trades_df=self.trades(pcode,index)
        prices_df=trades_df['price'].resample('1min',how='ohlc')
        vols_df=trades_df['volume'].resample('1min',how='sum')
        mins=pd.concat((prices_df,vols_df,),axis=1)
        if self.mins.has_key(pcode):
            pass
        else:
            self.mins[pcode]=mins
        return mins
