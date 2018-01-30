# -*- coding: utf-8 -*-

"""
В этом файле собран микс из наработок Гречко и моих + гибхабовских
Вычисляются характеристические функции, вводятся функции оценок опционов в моделях Мертона и Хестона,
калибруется модель Хестона.

По некоторому размышлению, я решил оставить этот файл без изменений, но разнести по разным файлам его содержимое
"""
from numpy import *
from scipy.optimize import fmin as sfmin
# фишка подхода в том, что параметры модели передаются в виде списка p (от "parameters")


def bs_characteristic_exp(u, p, r=0.0):
    """
     Характеристическая экспонента гауссовского процесса
     u - аргумент, который привычно видеть как ksi,
     p - параметры модели (здесь только один p[0] - волатильность),
     r - безрисковая ставка
    """
    return p[0]*p[0]/2.0*u*u-1j*u*(r-p[0]*p[0]/2.0)


def characteristic_func(char_exp, u, p, r, t):
    return exp(-t*char_exp(u, p, r))


def psi(char_exp, p, v, a, r=0.0, t=0.0):
    return exp(-r*t)*characteristic_func(char_exp, complex(v, -(a+1)), p, r, t)/complex(a*a+a-v*v, (2*a+1)*v)
    # альтернативно - return exp(-r*t)*char_func(v-1j*(a+1),p,r,t)/(a*a+a-v*v+1j*(2*a+1)*v)


def call_pricing(s, kmin, kmax, char_exp, p, r=0.0, t=0.0):
    # по формуле Карра-Мадана?
    b1 = abs(log(kmin/s))
    b2 = abs(log(kmax/s))
    b = b1
    if b2 > b1:
        b = b2
    n = 4096
    a = 1.5
    l = 2*b/n
    nu = pi/b
    d = zeros(n, dtype=complex)
    v = 0.0
    d1 = -1
    d2 = 1
    for i in range(n):
        d[i] = exp(1j*b*v)*psi(char_exp, p, v, a, r, t)*nu/3.0*(3.0+d1-d2)
        v += nu
        d1 *= -1
        d2 = 0
    c = fft.fft(d)
    ki = -b
    for i in range(n):
        k = s*exp(ki)
        c[i] = exp(-a*ki)/pi*c[i]*s
        print(k, ' ', c[i].real)
        ki += l


def call_pricing2(s, k, char_exp, p, r, t):
    # не очень ясно, по какой формуле
    n = 10000
    kl = log(k/s)
    sum = 0.0
    a = 1.5
    up = 30.0
    nu = up/n
    v = 0.0
    for i in range(n):
        sum += exp(-1j*v*kl)*psi(char_exp, p, v, a, r, t)*nu
        v += nu
    c = exp(-a*kl)/pi*sum*s
    return c.real


def put_pricing2(s, k, char_exp, p, r, t):
    # через паритет
    c = call_pricing2(s, k, char_exp, p, r, t)
    return c-s+k*exp(-r*t)


def merton_char_exp(u, p, r=0.0):
    # характеристическая экспонента для модели Мертона
    # p[0] - sigma>0, p[1] - lambda>0, p[2] - delta>0, p[3] - mu
    # Похоже, u = ksi, lj = complex(j)
    g0 = r-p[0]*p[0]/2.0+p[1]*(1-exp(p[2]*p[2]/2.0+p[3]))
    return p[0]*p[0]/2.0*u*u-1j*g0*u+p[1]*(1-exp(-p[2]*p[2]/2.0*u*u+1j*p[3]*u))


def kou_char_exp(u, p, r=0.0):
    # характеристическая экспонента для модели Коу
    # p[0] - sigma>0, p[1] - c+>0, p[2] - c->0, p[3] - l+>1, p[4] - l<0
    g0=r-p[0]*p[0]/2.0+p[1]/(1-p[3])+p[2]/(1-p[4])
    # print g0
    return p[0]*p[0]/2.0*u*u-1j*g0*u+1j*p[1]*u/(1j*u-p[3])+1j*p[2]*u/(1j*u-p[4])


# def kou_char_exp2(u,p,r=0.0):
#     a=p[1]*1j*u*p[3]/(1-1j*u*p[3])+p[2]*1j*u*p[4]/(1+1j*u*p[4])
#     g0=r-p[1]/(1-p[3])-p[2]/(1+p[4])+p[1]+p[2]
#     return (g0-p[0]*p[0]/2.0)*1j*u-p[0]*p[0]/2.0*u*u+a


# (k, option_price, put/call, t), список кортежей
# здесь  true для колла, false для пута
opts = [(160000.0, 27565.0, False, 0.22666152778261167), (135000.0, 5195.0, True, 0.26456776846204105),
      (135000.0, 7565.0, False, 0.2641434669240461), (110000.0, 23960.0, True, 0.37440672424472254),
      (115000.0, 19085.0, True, 0.3187342451566085), (115000.0, 1395.0, False, 0.3134406313934762),
      (120000.0, 15470.0, True, 0.335039269474388), (120000.0, 2290.0, False, 0.303147991328373),
      (100000.0, 33315.0, True, 0.42984421394326455), (100000.0, 305.0, False, 0.3618295814823113),
      (125000.0, 11145.0, True, 0.28968298757103794), (125000.0, 3530.0, False, 0.289925002660526),
      (130000.0, 7840.0, True, 0.27524345272031414), (130000.0, 5185.0, False, 0.27372703435372564),
      (155000.0, 22770.0, False, 0.2266502659261106), (140000.0, 3170.0, True, 0.25387291512296956),
      (140000.0, 10540.0, False, 0.2534113052757669), (145000.0, 1750.0, True, 0.24333086699998477),
      (145000.0, 13855.0, False, 0.227386877016284), (150000.0, 17915.0, False, 0.20483357470742397),
      (105000.0, 28330.0, True, 0.3716282062150906)]


def rmse(char_exp, p):
    """
    Это средняя квадратическая ошибка, в некотором смысле. Она применяется дальше при калибровке на сете опционов opts.
    Принимает любую характеристическую экспоненту, которую в состоянии обработать формула call_pricing2
    Внутри в качестве процентной ставки выбирается 0, в силу природы производных ЦБ
    """
    s = 132620.0
    t = 0.200093068239
    err = 0
    for opt in opts:
        op = 0
        if opt[2]:
            op = call_pricing2(s, opt[0], char_exp, p, 0, t)
        else:
            op = put_pricing2(s, opt[0], char_exp, p, 0, t)
        err += (op-opt[1])**2
    return sqrt(err/len(opts))


def rmse_merton(p):
    return rmse(merton_char_exp, p)


def rmse_kou(p):
    return rmse(kou_char_exp, p)


def sample_data():
    # sample market data
    x = [x.split() for x in open('marketdata.txt')]
    header = x[0]
    market_datas = []
    for market_data in x[1:]:
        market_datas.append([float(elem) for elem in market_data])
    return header, market_datas


def calibrate(init_val, market_datas):
    """
    parameter set p calibration

    Параметры:
    p[0] - sigma>0,
    p[1] - lambda>0,
    p[2] - delta>0,
    p[3] - mu

    market_datas - информация с рынка с полями
    s - цена базового актива (например, фьюч)
    k - страйк
    price - цена опциона по рынку (в некотором смысле)
    r - безрисковая процентная ставка
    t - срок до истечения

    Идея калибровки состоит в том, чтобы определить некоторую функцию ошибок и начальное приближение,
    затем применить некоторую оптимизирующую функцию (здесь используется алгоритм Нелдера-Мида)
    Возможная начальная догадка: [0.2, 1.0, 0.2, 0.2]
    """
    def error(x, market_datas):
        """
        Идея калибровки в минимизации функции ошибок,
        здесь используется наивный квадрат ошибки
        """
        sigma, lambda_m, delta, mu = x  # the name lambda_m is to avoid occasional use of the keyword
        result = 0.0
        for market_data in market_datas:
            s0, k, market_price, r, T = market_data
            merton_price = call_pricing2(s0, k, merton_char_exp, (sigma, lambda_m, delta, mu), r, t)
            result += (merton_price - market_price)**2
        print(sigma, lambda_m, delta, mu, 'sqr_error = ', result)
        return result
    opt = sfmin(error, init_val, args=(market_datas,), maxiter=20)
    return opt

if __name__ == '__main__':
    import time
    s = 132620.0
    # t=0.200093068239
    t = 0.2
    # vol=0.3618
    vol = 0.3
    # strike
    k = 100000.0

    l = 7.0
    lp = 5.0
    lm = -10.0
    p = 0.6  # похоже, в контексте она используется просто как волатильность модели Б-Ш

    time_start = time.clock()
    print(call_pricing2(s, k, merton_char_exp, (vol, 1.0, 0.2, 0.2), 0.0, t))
    time_elapsed = (time.clock() - time_start)
    print('time elapsed: %2f seconds' % time_elapsed)

    header, market_datas = sample_data()
    # Initialize vol=sigma, lambda, delta, mu
    init_val = [0.2, 1.0, 0.2, 0.2]
    print(market_datas[1])
    # calibration of parameters
    sigma, lambda_m, delta, mu = calibrate(init_val, market_datas)


    # print optimize.minimize(rmse_merton,[0.2,1.0,0.2,0.2],
    #   bounds=((0,None),(0,None,),(0,None),(None,None),),method='slsqp')
    # p=[0.2,0.5,1.4,1.0/2.0,1.0/3.0]
    # print optimize.minimize(rmse_kou,[0.4,0.3,0.7,1.0/2.0,1.0/3.0],
    #   bounds=((0,None),(0,None),(0,None),(None,None),(None,None)),
    #                        method='l-bfgs-b')

    # p1=option.euro_option_put_pricing(s,0,t,vol,0.0,k)
    # print p1
    # call_pricing(s,100.0,2500.0,bs_characteristic_exp,(vol,),0.0,t)
    # print call_pricing2(s,k,bs_characteristic_exp,(vol,),0.0,t)

    # print call_pricing2(s,k,kou_char_exp,(vol,l*p,l*(1-p),lp,lm),0.0,t)
    #
    # time_start = time.clock()
    # print optimize.minimize(rmse_kou, [vol, l * p, l * (1 - p), lp, lm],
    #                        bounds=((0, 1), (0.0001, 1), (0.0001, 1), (1.0, 1), (-1, 0)),
    #                        method='slsqp')
    # time_elapsed = (time.clock() - time_start)
    # print time_elapsed
    #
    # print kou_char_exp2(-1,p)
    # print kou_char_exp2(1,p)

    # print call_pricing2(s,k,kou_char_exp2,p,0.0,t)
    # print put_pricing2(s,k,kou_char_exp2,p,0.0,t)

    # call_pricing(s,50000.0,150000.0,kou_char_exp,(vol,l*p,l*(1-p),lp,lm),0.0,t)
    # print call_pricing2(s,k,kou_char_exp,(vol,l*p,l*(1-p),lp,lm),0.0,t)
