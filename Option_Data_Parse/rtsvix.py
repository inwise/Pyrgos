# -*- coding: utf-8 -*-

import win32com.client
import datetime
from math import sqrt,log
import matplotlib.pyplot as plt
from numpy import *
from scipy.stats import norm
import quotes
import option

__author__ = 'Alexander Grechko'

base_p_code='RTSI-9.14'


def add_to_option_data(data):
    # конвертирует данные по опционам из формата альфа-директ
    option_dict=dict()
    for d in data:
        mat_date=datetime.datetime.strptime(d[0]+" 18:50", "%d.%m.%Y %H:%M")
        if mat_date<datetime.datetime.today():
            continue
        for i in range(1,len(d)):
            if d[i]=='':
                d[i]='0'
        # o=option_data(d[0],float(d[1]),float(d[2]),float(d[3]),float(d[4]),float(d[5]),float(d[6]),float(d[7]))
        o = option_data(mat_date, float(d[1]), float(d[2]), float(d[3]),
                        float(d[4]), float(d[5]), float(d[6]), float(d[7]))
        if not option_dict.has_key(mat_date):
            option_dict[mat_date] = dict()
        option_dict[mat_date][o.strike] = o
    return option_dict


class option_data:
    # класс данных опциона

    def __init__(self, mat_date, strike, put_last_price, put_buy, put_sell, call_last_price, call_buy, call_sell):
        self.strike = strike
        self.put_last_price = put_last_price
        self.put_buy = put_buy
        self.put_sell = put_sell
        self.call_last_price = call_last_price
        self.call_buy = call_buy
        self.call_sell = call_sell
        self.mat_date = mat_date

    def check_validity(self):
        # проверяет есть ли все котировки bid, ask
        if self.put_buy == 0 or self.put_sell == 0 or self.call_buy == 0 or self.call_sell == 0:
            return False
        return True


def get_valid_options(opt_dict):
    # определяет K0 и валидные колы и путы для заданного набора опционов с одинаковой датой исполнения
    strikes = opt_dict.keys()
    strikes.sort()
    min = -1
    k0 = 0
    for strike in strikes:
        opt_data = opt_dict[strike]
        if opt_data.call_last_price == 0 or opt_data.put_last_price == 0:
            continue
        dif = abs(opt_data.call_last_price-opt_data.put_last_price)
        if dif <= min or min == -1:
            min = dif
            k0 = strike
    f = k0+opt_dict[k0].call_last_price-opt_dict[k0].put_last_price
    #f=candle.close
    c = 2
    put_options = list()
    call_options = list()
    for strike in strikes:
        opt_data = opt_dict[strike]
        if strike <= k0:
            if opt_data.put_sell == 0 or opt_data.put_buy == 0:
                continue
            if opt_data.put_sell/opt_data.put_buy<c:
                put_options.append((strike, (opt_data.put_sell+opt_data.put_buy)/2,
                                    opt_data.put_sell, opt_data.put_buy, ))
        else:
            if opt_data.call_sell == 0 or opt_data.call_buy == 0:
                continue
            if opt_data.call_sell/opt_data.call_buy<c:
                call_options.append((strike, (opt_data.call_sell+opt_data.call_buy)/2, opt_data.call_sell,
                                     opt_data.call_buy,))
    return k0, f, put_options, call_options

def get_d2(k, t, vol):
    #вычисляет d2
    a = vol*sqrt(t)
    d2 = -k/a-a/2
    return d2

def get_implied_volatilities(mat_date, valid_options, date=None):
    # считает implied volatility для полученных опционов, оставляя интервалы где d2 монотонно убывает
    f = valid_options[1]
    t = option.get_t(mat_date,date)
    x = list()
    y = list()
    for put_option in reversed(valid_options[2]):
        vol = option.euro_option_put_volatility(f, 0, t, 0, put_option[0], put_option[1])
        k = log(put_option[0]/f)
        d2 = get_d2(k, t, vol)
        if len(x) > 0:
            if d2 < x[-1]:
                break
        x.append(d2)
        y.append(vol*vol)
    x.reverse()
    y.reverse()
    for call_option in valid_options[3]:
        vol=option.euro_option_call_volatility(f, 0, t, 0, call_option[0], call_option[1])
        k=log(call_option[0]/f)
        d2=get_d2(k, t, vol)
        if len(x) > 0:
            if d2 > x[-1]:
                break
        x.append(d2)
        y.append(vol*vol)
    x.reverse()
    y.reverse()
    return x, y,


def get_distance(p1,p2):
    # вычисляет расстояние между точками p1 и p2
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def get_dy(points):
    # вычисляет производную функции y в каждой точке
    x, y = points
    dy = zeros(len(x))
    if len(x) <= 2:
        return dy
    l1=get_distance((x[0], y[0]), (x[1], y[1]))
    for j in range(1, len(x)-1):
        l2=get_distance((x[j], y[j]), (x[j+1], y[j+1]))
        dy[j]=-((x[j+1]-x[j])/l2-(x[j]-x[j-1])/l1)/((y[j+1]-y[j])/l2-(y[j]-y[j-1])/l1)
        l1=l2
    return dy


def get_cubic_polynomial_approximation(points):
    #строит апроксимацию кубическими полиномами точек (x,y)
    x, y = points
    dy = get_dy((x, y,))
    m = len(x)
    a = y[:-1]
    b = dy
    c = empty(m-1)
    d = empty(m-1)
    for j in range(m-1):
        dxj = x[j+1]-x[j]
        dyj = y[j+1]-y[j]
        c[j] = (3*dyj-dxj*dy[j+1]-2*dxj*dy[j])/(dxj*dxj)
        d[j] = (dyj-dy[j]*dxj-c[j]*dxj*dxj)/(dxj**3)
    return a, b, c, d,


def get_approximation_values(points, approx, bounds=None):

    # строит значения апроксимированной фукнции
    x, y=points
    m = len(x)
    a, b, c, d = approx
    xmin = x[0]
    xmax = x[m-1]
    if bounds != None:
        xmin, xmax = bounds
    if xmin >= xmax:
        return ([], [])
    n = 1000
    xstep=(xmax-xmin)/n
    xn=empty(n+1)
    yn=empty(n+1)
    i=0
    for j in range(n+1):
        xn[j]=xmin+j*xstep
        while (i<m):
            if xn[j]<=x[i]:
                break
            i+=1
        if i  == 0:
            yn[j]=y[0]
        elif i  == m:
            yn[j]=y[m-1]
        else:
            yn[j]=a[i-1]+b[i-1]*(xn[j]-x[i-1])+c[i-1]*(xn[j]-x[i-1])**2+d[i-1]*(xn[j]-x[i-1])**3
    return (xn, yn, )

def integrate_approximation_func(points, pols):
    #интегрирует апроксимированную функцию
    x, y=points
    a, b, c, d=pols
    m=len(x)
    cdf_x1=norm.cdf(x[0])
    sum=y[0]*cdf_x1
    pdf_x1=norm.pdf(x[0])
    for j in range(m-1):
        #cdf_x1=norm.cdf(x[j])
        cdf_x2=norm.cdf(x[j+1])
        #pdf_x1=norm.pdf(x[j])
        pdf_x2=norm.pdf(x[j+1])
        ka=cdf_x2-cdf_x1
        kb=-(pdf_x2-pdf_x1)-x[j]*(cdf_x2-cdf_x1)
        kc=-(x[j+1]*pdf_x2-x[j]*pdf_x1)+2*x[j]*(pdf_x2-pdf_x1)+(1+x[j]**2)*(cdf_x2-cdf_x1)
        kd=(1-x[j+1]**2)*pdf_x2-(1-x[j]**2)*pdf_x1+3*x[j]*(x[j+1]*pdf_x2-x[j]*pdf_x1)-3*(1+x[j]**2)*(pdf_x2-pdf_x1)-x[j]*(3+x[j]**2)*(cdf_x2-cdf_x1)
        sum+=a[j]*ka+b[j]*kb+c[j]*kc+d[j]*kd
        cdf_x1=cdf_x2
        pdf_x1=pdf_x2
    sum+=y[m-1]*(1-cdf_x1)
    return sum

def get_current_rtsi(ad):
    res=ad.GetLocalDBData('fin_info', 'last_price,  last_update_time,  last_update_date,  sell,  buy', "p_code='%s'"%(base_p_code, ))
    if res  == "":
        print "Error ", ad.LastResultMsg
        return None
    data=quotes.ad_parse_result(res)[0]
    value=float(data[0])
    date=datetime.datetime.strptime(data[2]+" "+data[1], "%d.%m.%Y %H:%M:%S")
    return (value, date, float(data[3]), float(data[4]))
    '''
    today=datetime.datetime.today()
    candle=quotes.load_quotes_from_ad_server(ad, 'FORTS', base_p_code, 0, today-datetime.timedelta(days=3), today)[-1]
    return candle.close
    '''

def plot_implied_volatility_smile(option_dict, rtsi, sub_plot = None):
    #рисует улыбку волатильности
    mult=1
    if sub_plot   ==  None:
        sub_plot=plt
    mat_dates=option_dict.keys()
    mat_dates.sort()
    for mat_date in mat_dates:
        t=option.get_t(mat_date)
        opt_dict=option_dict[mat_date]
        strikes=opt_dict.keys()
        strikes.sort()
        vols=list()
        res_strikes=list()
        for strike in strikes:
            opt_data=opt_dict[strike]
            price=0
            if strike<=rtsi:
                if opt_data.put_last_price != 0:
                    price=opt_data.put_last_price
                elif opt_data.put_buy != 0 and opt_data.put_sell != 0:
                    price=(opt_data.put_buy+opt_data.put_sell)/2
                if price != 0:
                    vol=option.euro_option_put_volatility(rtsi, 0, t, 0, strike, price)*mult
                    vols.append(vol)
                    res_strikes.append(strike)
            else:
                if opt_data.call_last_price != 0:
                    price=opt_data.call_last_price
                elif opt_data.call_buy != 0 and opt_data.call_sell != 0:
                    price=(opt_data.call_buy+opt_data.call_sell)/2
                if price != 0:
                    vol=option.euro_option_call_volatility(rtsi, 0, t, 0, strike, price)*mult
                    vols.append(vol)
                    res_strikes.append(strike)
        #sub_plot.xlabel("K")
        #sub_plot.set_xlabel("K")
        #sub_plot.ylabel(r'$\sigma$')
        #sub_plot.set_ylabel(r'$\sigma$')
        sub_plot.plot(res_strikes, vols, label=mat_date.strftime("%d.%m.%Y"))
        #sub_plot.title("volatility smile")
        #sub_plot.show()
    sub_plot.legend()

def plot_implied_volatility(option_dict, rtsi):
    #рисует графики подразумеваемой волатильности для call и put
    mult=sqrt(252)
    for mat_date in option_dict.keys():
        t=option.get_t(mat_date)
        put_strikes=list()
        call_strikes=list()
        put_prices=list()
        call_prices=list()
        put_vol=list()
        call_vol=list()
        opt_dict=option_dict[mat_date]
        strikes=opt_dict.keys()
        strikes.sort()
        for strike in strikes:
            opt_data=opt_dict[strike]
            if opt_data.put_last_price != 0:
                put_prices.append(opt_data.put_last_price)
                vol=option.euro_option_put_volatility(rtsi, 0, t, 0, strike, opt_data.put_last_price)*mult
                put_vol.append(vol)
                put_strikes.append(strike)
            if opt_data.call_last_price != 0:
                call_prices.append(opt_data.call_last_price)
                vol=option.euro_option_call_volatility(rtsi, 0, t, 0, strike, opt_data.call_last_price)*mult
                call_vol.append(vol)
                call_strikes.append(strike)
        plt.subplot(2, 2, 1)
        plt.plot(put_strikes, put_prices)
        plt.title(u"put")
        plt.subplot(2, 2, 2)
        plt.plot(call_strikes, call_prices)
        plt.title(u"call")
        plt.subplot(2, 2, 3)
        plt.plot(put_strikes, put_vol)
        plt.subplot(2, 2, 4)
        plt.plot(call_strikes, call_vol)
        plt.show()

def get_rtsvix(points, pols):
    return sqrt(integrate_approximation_func(points, pols))*100

def example():
    x=[2.322589, 1.737578, 1.597871, 1.428667, 1.243389, 1.054255, 0.833485, 0.595460, 0.347682, 0.077152, -0.211813, -0.516513,
       -0.820640, -1.128248, -1.410956, -1.678436, -1.941339, -2.158142, -2.3338]
    y=[0.1953966, 0.1401579, 0.1247173, 0.1129279, 0.1025435, 0.0913947, 0.0835569, 0.0768361, 0.0690620, 0.0627555, 0.0586251,
       0.0540715, 0.0523597, 0.0506391, 0.0510783, 0.0519399, 0.0524815, 0.0549685, 0.0588631]
    x.reverse()
    y.reverse()
    return (x, y, )

def get_option_dict(ad):
    #запрос доски опционов
    res=ad.GetLocalDBData('option_board', 'mat_date,  strike,  put_last_price,  put_buy,  put_sell,  call_last_price,  call_buy,  call_sell',
                          "base_p_code='%s'"%(base_p_code, ))
    if res  == "":
        print "Error ", ad.LastResultMsg
    data=quotes.ad_parse_result(res)
    option_dict=add_to_option_data(data)
    return option_dict

def plot_rtsvix():
    #рисует индекс волатильности
    ad = win32com.client.Dispatch("ADLite.AlfaDirect")
    if not ad.Connected:
        ad.UserName="agrechko"
        ad.Password="s8x1g5"
        ad.Connected=True
    option_dict=get_option_dict(ad)
    for mat_date in option_dict:
        opt_dict=option_dict[mat_date]
        valid_options=get_valid_options(opt_dict)
        print valid_options[1]
        x, y=get_implied_volatilities(mat_date, valid_options)
        pols=get_cubic_polynomial_approximation((x, y, ))
        xn, yn=get_approximation_values((x, y, ), pols, (-3.0, 3.0, ))
        print len(x)
        print get_rtsvix((x, y, ), pols)
        plt.subplot(2, 1, 1)
        plt.xlabel('d2')
        print "OK"
        plt.ylabel(r'$\sigma^2$')
        plt.plot(x, y)
        plt.subplot(2, 1, 2)
        plt.xlabel('d2')
        plt.ylabel(r'$\sigma^2$')
        plt.plot(xn, yn)
        plt.show()

def get_last_rtsvx_info(ad):

    # возвращает последнюю свечу по индексу волатильности РТС
    today=datetime.datetime.today()
    # candle=quotes.load_quotes_from_ad_server(ad, 'INDEX', 'RTSVX', 0, today-datetime.timedelta(days=3), today)[-1]
    res=ad.GetLocalDBData('fin_info', 'last_price,  last_update_time,  last_update_date', "p_code='%s'"%('RTSVX', ))
    if res == "":
        print "Error ", ad.LastResultMsg
        return None
    data=quotes.ad_parse_result(res)[0]
    value=data[0]
    date=datetime.datetime.strptime(data[2]+" "+data[1], "%d.%m.%Y %H:%M:%S")
    return value, date,

def valid_options_from_ad(ad):
    option_dict=get_option_dict(ad)
    mat_date=min(option_dict.keys())
    opt_dict=option_dict[mat_date]
    valid_options=get_valid_options(opt_dict)
    return mat_date, valid_options,

def get_current_implied_volatilities(ad):
    try:
        value, date=get_last_rtsvx_info(ad)
    except TypeError:
        return None
    mat_date, valid_options=valid_options_from_ad(ad)
    x, y=get_implied_volatilities(mat_date, valid_options, date)
    return (x, y, ), date, mat_date

def rtsvix(ad):

    # считает и возвращает индекс волатильности
    points, date, mat_date=get_current_implied_volatilities(ad)
    x, y=points
    if len(x)<=1:
        return None
    pols=get_cubic_polynomial_approximation((x, y, ))
    return (get_rtsvix((x, y, ), pols), date, )

def rub_to_points(rub):
    #переводит рубли в пункты
    return int(rub/67.0*50.0)

def test_arbitrage_oppportunities(option_dict, rtsi, rtsi_sell, rtsi_buy):
    #проверяет выполняется ли условия отсутствия арбитража у опционов
    global sended
    delta=int(8.0/37.0*50.0)
    mat_dates=option_dict.keys()
    if len(mat_dates)<1:
        return
    mat_dates.sort()
    mat_date=mat_dates[0]
    opt_dict=option_dict[mat_date]
    strikes=opt_dict.keys()
    strikes.sort()
    if len(strikes)<3:
        return
    now_str=datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    for i in range(len(strikes)):
        opt_data_i=opt_dict[strikes[i]]

        #проверка условий call-put паритета
        if opt_data_i.call_buy != 0 and opt_data_i.put_sell != 0:
            if opt_data_i.call_buy-opt_data_i.put_sell>rtsi_sell-strikes[i]+rub_to_points(20):
                print "%s arbitrage call-put parity C-P>S-K strike = %g"%(now_str, strikes[i])
        if opt_data_i.call_sell != 0 and opt_data_i.put_buy != 0:
            if opt_data_i.call_sell-opt_data_i.put_buy+rub_to_points(20)<rtsi_buy-strikes[i]:
                print "%s arbitrage call-put parity C-P<S-K strike = %g"%(now_str, strikes[i])

        #проверка условий раннего исполнения
        if strikes[i]<rtsi:
            if opt_data_i.call_sell != 0:
                if opt_data_i.call_sell<rtsi-strikes[i]:
                    print "%s arbitrage early exercise call strike = %g"%(now_str, strikes[i], )
        else:
            if opt_data_i.put_sell != 0:
                if opt_data_i.put_sell<strikes[i]-rtsi:
                    print "%s arbitrage early exercise put strike = %g"%(now_str, strikes[i], )

        #проверка условий монотонности
        if i>0:
            opt_data_i1=opt_dict[strikes[i-1]]
            if opt_data_i1.call_sell != 0 and opt_data_i.call_buy != 0:
                if opt_data_i1.call_sell+2*delta<=opt_data_i.call_buy:
                    print "%s arbitrage monotonic opportunity call strikes = %g,  %g"%(now_str, strikes[i-1], strikes[i], )
            if opt_data_i.put_sell != 0 and opt_data_i1.put_buy != 0:
                if opt_data_i1.put_buy>=opt_data_i.put_sell+2*delta:
                    print "%s arbitrage monotonic opportunity put strikes = %g,  %g"%(now_str, strikes[i-1], strikes[i], )

        #проверка условий выпуклости
        if i>1:
            opt_data_i2=opt_dict[strikes[i-2]]
            if opt_data_i1.call_buy != 0 and opt_data_i2.call_sell != 0 and opt_data_i.call_sell != 0:
                if opt_data_i2.call_sell+opt_data_i.call_sell+rub_to_points(24)<=2*opt_data_i1.call_buy:
                    print "%s arbitrage convex opportunity call strike = %g"%(now_str, strikes[i-1], )
                    '''
                    if (not sended) and valid_time():
                        pcode_i=get_option_code(opt_data_i)
                        pcode_i1=get_option_code(opt_data_i1)
                        pcode_i2=get_option_code(opt_data_i2)
                        o=order(True, 0, opt_data_i.call_sell, 1)
                        o1=order(False, 0, opt_data_i1.call_buy, 2)
                        o2=order(True, 0, opt_data_i2.call_sell, 1)
                        eng.set_order(pcode_i, o)
                        eng.set_order(pcode_i2, o2)
                        eng.set_order(pcode_i1, o1)
                        sended=True
                    '''
            if opt_data_i1.put_buy != 0 and opt_data_i2.put_sell != 0 and opt_data_i.put_sell != 0:
                if opt_data_i2.put_sell+opt_data_i.put_sell+rub_to_points(24)<=2*opt_data_i1.put_buy:
                    print "%s arbitrage convex opportunity put strike = %g"%(now_str, strikes[i-1], )
                    '''
                    if (not sended) and valid_time():
                        pcode_i=get_option_code(opt_data_i, False)
                        pcode_i1=get_option_code(opt_data_i1, False)
                        pcode_i2=get_option_code(opt_data_i2, False)
                        o=order(True, 0, opt_data_i.put_sell, 1)
                        o1=order(False, 0, opt_data_i1.put_buy, 2)
                        o2=order(True, 0, opt_data_i2.put_sell, 1)
                        eng.set_order(pcode_i, o)
                        eng.set_order(pcode_i2, o2)
                        eng.set_order(pcode_i1, o1)
                        sended=True
                    '''
