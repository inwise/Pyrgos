# -*- coding: utf-8 -*-

__author__ = 'Alexander Grechko'

import urllib
import datetime
import math
from numpy import *
from matplotlib.dates import date2num
import copy
import pandas as pd
from StringIO import StringIO
from pandas.tseries.offsets import BDay


class history:
    # история котировок
    def __init__(self, candles):
        n = len(candles)
        self.highs = empty(n)
        self.lows = empty(n)
        self.opens = empty(n)
        self.closes = empty(n)
        for i in range(n):
            self.highs[i] = candles[i].high
            self.lows[i] = candles[i].low
            self.opens[i] = candles[i].open
            self.closes[i] = candles[i].close


class candle:
    # класс свечи
    def __init__(self, dt, open, high, low, close, volume=-1, timeframe=1):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.date = dt
        self.timeframe = timeframe

    def copy_close_only(self):
        candle = copy.copy(self)
        candle.open = candle.close
        candle.high = candle.close
        candle.low = candle.close
        return candle

    def __str__(self):
        #строковое представление объекта
        s = "%s open=%g high=%g low=%g close=%g" % (
            self.date.strftime("%Y/%m/%d %H:%M:%S"), self.open, self.high, self.low, self.close)
        return s


def load_quotes_from_alor(board, ticker, period, from_time, to_time):
    # загрузка котировок с сервер Алор
    params = urllib.urlencode({'board': board, 'ticker': ticker, 'period': period,
                               'from': from_time, 'to': to_time, 'bars': 1000})
    f = urllib.urlopen("http://history.alor.ru/?%s" % params)
    candles = list()
    for line in f.readlines():
        a = line.split()
        if len(a) == 7:
            candles.append(
                candle(datetime.strptime(a[0] + " " + a[1], "%Y-%m-%d %H:%M:%S"), float(a[2]), float(a[3]), float(a[4]),
                       float(a[5]), int(a[6])))
    candles.sort(key=lambda x: x.date)
    return candles


def load_quotes_from_finam_file(fn):
    # загрузка котировок из файла в формате финам
    f = open(fn)
    candles = list()
    f.readline()
    for line in f:
        s = line.split(',')
        timeframe = 1440
        if s[1] != "D":
            timeframe = int(s[1])
        candles.append(candle(datetime.datetime.strptime(s[2] + " " + s[3], "%Y%m%d %H%M%S"), float(s[4]), float(s[5]),
                              float(s[6]), float(s[7]), -1, timeframe))
    f.close()
    return candles


def load_quotes_from_file(fn):
    # загрузка котировок из файла в формате финам
    f = open(fn)
    candles = list()
    f.readline()
    timeframe = 1440
    for line in f:
        s = line.split(',')
        candles.append(
            candle(datetime.datetime.strptime(s[0] + " " + s[1], "%d.%m.%Y %H:%M:%S"), float(s[2]), float(s[3]),
                   float(s[4]), float(s[5]), float(s[6]), timeframe))
    f.close()
    return candles


def fill_holes(candles):
    # заполняем дыры в котировках
    tf = candles[0].timeframe
    if tf == 1440:
        return candles
    ncandles = [candles[0]]
    for i in range(1, len(candles)):
        if candles[i].date.date() == candles[i - 1].date.date():
            #внутри одного дня
            dt = candles[i].date - candles[i - 1].date
            k = dt.seconds / 60 / tf
            for j in range(k - 1):
                c = candles[i - 1].copy_close_only()
                c.date += (j + 1) * datetime.timedelta(minutes=1) * tf
                ncandles.append(c)
        else:
            #разные дни
            a = (
                    datetime.datetime.combine((candles[i - 1].date + datetime.timedelta(days=1)).date(),
                                              datetime.time(0)) -
                    candles[i - 1].date).seconds / 60 / tf
            b = (candles[i].date - datetime.datetime.combine(candles[i].date.date(),
                                                             datetime.time(hour=10))).seconds / 60 / tf
            k = a + b
            for j in range(k - 1):
                ncandles.append(candles[i - 1].copy_close_only())
        ncandles.append(candles[i])
    return ncandles


def get_rates_of_return(candles, holes=True, disc=False):
    # возвращает доходности активов по истории, итоговый список содержит на один элемент меньше
    n = len(candles)
    rates = list()
    for i in range(1, n):
        if candles[i].timeframe != 1440 and holes:
            #если не дневки заполняем дыры в котировках
            if candles[i].date.date() == candles[i - 1].date.date():
                #внутри одного дня
                dt = candles[i].date - candles[i - 1].date
                k = dt.seconds / 60 / candles[i].timeframe
                for j in range(k - 1):
                    rates.append(0.0)
            else:
                #разные дни
                a = (datetime.datetime.combine((candles[i - 1].date + datetime.timedelta(days=1)).date(),
                                               datetime.time(0)) - candles[i - 1].date).seconds / 60 / candles[
                        i].timeframe
                b = (candles[i].date - datetime.datetime.combine(candles[i].date.date(),
                                                                 datetime.time(hour=10))).seconds / 60 / candles[i].timeframe
                k = a + b
                for j in range(k - 1):
                    rates.append(0.0)
        if disc:
            #rates.append(candles[i].close-candles[i-1].close)
            rates.append((candles[i].close - candles[i - 1].close) / candles[i - 1].close)
        else:
            rates.append(math.log(candles[i].close / candles[i - 1].close))
    return array(rates)


def ad_parse_result(res):
    # парсит результат от терминала альфа-директ и возвращает данные в виде списка
    data = list()
    for s in res.split("\n"):
        a = s.split('|')
        if len(a) < 2:
            break
        data.append(a[:-1])
    return data


def to_float(a):
    # преобразование записи из базы ad в число
    return float(a.replace(',', '.'))


def fininfo_to_candles(res):
    # данные полученные от АлфаДирект преобразуем в список candle
    candles = list()
    for s in res.split("\n"):
        a = s.split('|')
        if len(a) < 6:
            break
        try:
            dt = datetime.datetime.strptime(a[0], "%d/%m/%Y %H:%M:%S")
        except ValueError:
            dt = datetime.datetime.strptime(a[0], "%d/%m/%Y")
        candles.append(candle(dt, to_float(a[1]), to_float(a[2]), to_float(a[3]), to_float(a[4]), to_float(a[5]), 1))
    return candles


def load_quotes_from_ad_server(ad, PlaceCode, pcode, period, from_time, to_time):
    res = ad.GetArchiveFinInfo(PlaceCode, pcode, period, from_time, to_time, 3, 120)
    if res == "":
        print "Error ", ad.LastResultMsg
        return
    return fininfo_to_candles(res)


def load_quotes_from_ad_db(ad, PlaceCode, pcode, period, from_time, to_time):
    res = ad.GetArchiveFinInfoFromDB(PlaceCode, pcode, period, from_time, to_time)
    if res == "":
        print "Error ", ad.LastResultMsg
        return
    return fininfo_to_candles(res)


def get_opens(candles):
    opens = list()
    for candle in candles:
        opens.append(candle.open)
    return array(opens)


def get_closes(candles):
    closes = list()
    for candle in candles:
        closes.append(candle.close)
    return array(closes)


def get_highs(candles):
    highs = list()
    for candle in candles:
        highs.append(candle.high)
    return array(highs)


def get_lows(candles):
    lows = list()
    for candle in candles:
        lows.append(candle.low)
    return array(lows)


def get_matplotlib_quotes(candles):
    qs = list()
    for candle in candles:
        qs.append((date2num(candle.date), candle.open, candle.close, candle.high, candle.low))
    return qs


def get_matplotlib_quotes_from_df(df):
    df = df.reset_index()
    df['datetime2'] = df['datetime'].apply(lambda date: date2num(date.to_pydatetime()))
    return df[['datetime2', 'open', 'close', 'high', 'low']].values


def load_quotes_from_ad_df(ad, PlaceCode, pcode, period, from_time, to_time, only_from_server = False):
    # загружает котировки из альфа-директ в объект pandas.DataFrame
    res = ad.GetArchiveFinInfo(PlaceCode, pcode, period, from_time, to_time, 2, 10)
    if res == "":
        #msg = "Error " + ad.LastResultMsg
        #print msg
        #sms_alert(msg)
	if not only_from_server:
            res = ad.GetArchiveFinInfoFromDB(PlaceCode, pcode, period, from_time, to_time)
        if res == "":
            return None
        else:
            print res
            print "Data from local DB"
    names = ('datetime', 'open', 'high', 'low', 'close', 'volume', 'n',)
    types = {'open': float64, 'high': float64, 'low': float64, 'close': float64, 'volume': int64}
    df = pd.read_csv(StringIO(res), delimiter='|', header=None, names=names, index_col=0, usecols=[0, 1, 2, 3, 4, 5],
                     parse_dates=True, dayfirst=True, infer_datetime_format=True, dtype=types, decimal=',')
    return df


def quarter_futures_exp_dates(y=datetime.datetime.today().year):
    # возвращает даты экспирации квартальных фьючерсов текущего года
    dates = list()
    for m in range(3, 15, 3):
        dates.append((datetime.datetime(y, m, 15) + BDay(0)).date())
    return dates


assets = ('RTSI', 'SBER', 'USD', 'BRENT',)
assets_dict = dict()
for asset in assets:
    assets_dict[asset] = 1

def future_code(asset, today = None):
    # возвращает код фьючерса
    #if not assets_dict.has_key(asset):
    #    return asset, None, None,
    if today is None:
        today = datetime.datetime.today()
    exp_dates = quarter_futures_exp_dates(today.year)
    y = today.year - 2000
    if today < exp_dates[0]:
        return ("%s-3.%d" % (asset, y,), quarter_futures_exp_dates(today.year - 1)[3], "%s-12.%d" % (asset, y - 1),)
    for i in range(1, 4):
        if today >= exp_dates[i - 1] and today < exp_dates[i]:
            return ("%s-%d.%d" % (asset, exp_dates[i].month, y,), exp_dates[i - 1],
                    "%s-%d.%d" % (asset, exp_dates[i - 1].month, y,),)
    return ("%s-3.%d" % (asset, y + 1,), exp_dates[3], "%s-12.%d" % (asset, y,))


def ad_result_to_float(res):
    # преобразует первую строку res в список float
    return [to_float(a) for a in ad_parse_result(res)[0]]


option_expiration_dates = ('2016-01-15','2016-01-20','2016-01-21','2016-02-01','2016-02-17','2016-02-18',)
option_expiration_dates_dict = dict()
for date in option_expiration_dates:
    option_expiration_dates_dict[date] = 1


def valid_time(dt=datetime.datetime.today()):
    # определяет dt - торговое время или нет
    #if dt.weekday() >= 5:
    #    return False
    if dt.hour < 10:
        return False
    if dt.hour == 23 and dt.minute >= 50:
        return False
    if dt.hour == 14 and dt.minute < 3:
        return False
    if dt.hour == 18 and dt.minute > 45:
        return False
    if option_expiration_dates_dict.has_key(dt.strftime("%Y-%m-%d")) and dt.hour == 19 and dt.minute < 5:
        return
    return True


def load_quotes_from_hdf_store(store, place_code, pcode, from_time=None, to_time=None):
    # загружает котировки из hdf файла
    key = place_code + '/' + pcode
    cond = list()
    if from_time is not None:
        cond.append(pd.Term('index', '>=', pd.Timestamp(from_time)))
    if to_time is not None:
        cond.append(pd.Term('index', '<=', pd.Timestamp(to_time)))
    return store.select(key, cond)

PlaceCode = "FORTS"

def load_future_quotes_from_hdf_store(store, asset_code, from_time=None, to_time=None):
    # загружает котировки фьючерса по активу из hdf файла, допускается использовать склеенный фьючерс, проводит склейку
    pcode, prev_exp_date, pcode_prev = future_code(asset_code, to_time)
    #print pcode
    pcode_last = pcode
    df = None
    to = to_time
    while from_time < prev_exp_date:
        tdf = load_quotes_from_hdf_store(store, PlaceCode, pcode, prev_exp_date, to)
        if df is None:
            df = tdf
        else:
            df = pd.concat((tdf, df,), verify_integrity=True)
        to = prev_exp_date - datetime.timedelta(minutes=1)
        pcode, prev_exp_date, pcode_prev = future_code(asset_code, to)
    tdf = load_quotes_from_hdf_store(store, PlaceCode, pcode, from_time, to)
    if df is None:
        df = tdf
    else:
        df = pd.concat((tdf, df,), verify_integrity=True)
    return df

def load_quotes_from_finam_file_df(fn):
    #загружает котировки из файла finam в формате pandas.DataFrame
    f = open(fn)
    names = ('date','time','open', 'high', 'low', 'close', 'volume',)
    types = {'open': float64, 'high': float64, 'low': float64, 'close': float64, 'volume': int64}
    df = pd.read_csv(f, parse_dates={'datetime': [0,1]}, skiprows=1, names=names, dtype=types, index_col=0)
    f.close()
    return df

def rates_of_return(x):
    #возвращает доходности, x - логарифм цен
    return (x-x.shift(1))[1:]

def log_price(df):
    #возвращает логарифм доходностей цен закрытия
    return log(df.close)
