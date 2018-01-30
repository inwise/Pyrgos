# -*- coding: utf-8 -*-
import heston.gh_heston as heston
from scipy.stats import uniform
from scipy.optimize import fmin
from merton.simple_datafeed import sample_dreamfile, sample_data


def error(x, market_datas):
    """
    Чтобы применять калибровку или подбирать нечто,
    нужно использовать некоторую функцию ошибок.
    Используется средняя квадратическая ошибка.

    :param x: - значения параметров модели Хестона, в порядке: kappa, theta, sigma, rho, v0
    :param market_datas: - рыночные данные, содержащие S0, K, put/call type, Option_Price, r, T
    :return: значение ошибки
    """

    kappa, theta, sigma, rho, v0 = x
    result = 0.0
    for market_data in market_datas:
        s0, k, put_or_call, market_price, r, T = market_data
        if put_or_call == 'c' or put_or_call == 'C':
            heston_price = heston.call_price(kappa, theta, sigma, rho, v0, r, T, s0, k)
        elif put_or_call == 'p' or put_or_call == 'P':
            heston_price = heston.put_price(kappa, theta, sigma, rho, v0, r, T, s0, k)
        else:
            print('Unknown option type')
            return 0.0
        result += (heston_price - market_price) ** 2
    result /= len(market_datas)
    # print(kappa, theta, sigma, rho, v0, 'mean_sqr_error = ', result)
    return result


def make_initial_parameters_guess(market_datas, trials=50):
    """
    Подбор начального приближения для калибровки модели Хестона

    К сожалению, генетические алгоритмы показали себя весьма жадными до времени исполнения в этих задачах.
    Полный перебор поверхностей занимал часы, калибровку для российского рынка в динамике за обозримое время
    сделать не получалось.

    Поэтому здесь мы используем наиболее примитивный вариант Монте-Карло, который занимает не так много времени:
    мы ищем хорошее начальное приближение посредством угадывания.

    Угадывание обеспечивается использованием равномерного распределения
    uniform.rvs(loc=-1, scale=2)
    в диапазоне (loc, loc+scale).

    Для всех параметров модели Хестона рамки изменения разные.

    :param market_datas: - рыночные данные, содержащие S0, K, Call_Price,r ,T
    :param trials: - количество точек для угадывания
    :return: [наименьшая из полученных ошибок, параметры, на которых она достигается]
    Параметры возвращаются в порядке kappa, theta, sigma, rho, v0
    """
    all_parameters = {}

    for _ in range(trials):
        kappa = uniform.rvs(loc=0, scale=2)
        theta = uniform.rvs(loc=0, scale=2)
        sigma = uniform.rvs(loc=0, scale=2)
        rho = uniform.rvs(loc=-1, scale=2)
        v0 = uniform.rvs(loc=0, scale=2)
        x = (kappa, theta, sigma, rho, v0)
        all_parameters[error(x, market_datas)] = x
    errors = list(all_parameters.keys())
    errors.sort(key=lambda er: float(er))
    best_error = errors[0]
    best_parameters = all_parameters[best_error]
    return best_error, best_parameters


def calibrate(init_val, market_datas,  max_iterations=40):
    """
    parameter set p calibration в модели Хестона

    Параметры:
    p[0]: kappa - +-inf
    p[1]: theta - +-inf
    p[2]: sigma - sigma > 0,
    p[3]: rho - (-1,1)
    p[4]: v0 > 0

    market_datas - информация с рынка с полями
    s - цена базового актива (например, фьюч)
    k - страйк
    call/put type - тип опциона
    price - цена опциона по рынку (в некотором смысле)
    r - безрисковая процентная ставка
    t - срок до истечения

    Идея калибровки состоит в том, чтобы определить некоторую функцию ошибок и начальное приближение,
    затем применить некоторую оптимизирующую функцию (здесь используется алгоритм Нелдера-Мида)

    :return: (корень из среднеквадратической ошибки, набор параметров, на которых она достигается)
    параметры возвращаются в порядке kappa, theta, sigma, rho, v0
    """
    # parameter calibration(kappa, theta, sigma, rho, v0)
    opt = fmin(error, init_val, args=(market_datas,), maxiter=max_iterations, disp=0, ftol=20000)
    return error(opt, market_datas)**0.5, opt

if __name__ == '__main__':
    import time
    time_start = time.clock()
    # load market data
    header, market_datas = sample_dreamfile()
    print(sample_data()[1])
    # Initialize kappa, theta, sigma, rho, v0
    guesses = 2
    init_error, init_val = make_initial_parameters_guess(market_datas, trials=guesses)
    print('initials are', init_error, init_val)
    time_elapsed_estimating = time.clock() - time_start
    print('time elapsed %2f sec, guesses made: %d' % (time_elapsed_estimating, guesses))
    # calibration of parameters
    time_start = time.clock()
    optimum = calibrate(init_val, market_datas, max_iterations=2)
    (kappa, theta, sigma, rho, v0) = optimum[1]
    print('optimals are: kappa = %2f, theta = %2f, sigma = %2f, rho = %2f, v0 = %2f' % (kappa, theta, sigma, rho, v0))
    print('error in optimals is %2f' % optimum[0])
    time_elapsed_calibration = time.clock() - time_start
    print('time for calibration %2f sec' % time_elapsed_calibration)
    # ______________DRAWING________________
    import numpy as np
    import matplotlib.pyplot as plt

    market_prices = np.array([])
    heston_prices = np.array([])
    K = np.array([])
    for market_data in market_datas:
        s0, k, put_call_type, market_price, r, T = market_data
        heston_prices = np.append(heston_prices, heston.call_price(kappa, theta, sigma, rho, v0, r, T, s0, k))
        market_prices = np.append(market_prices, market_price)
        K = np.append(K, k)
    # plot result
    plt.plot(K, market_prices, 'g*', K, heston_prices, 'b')
    plt.xlabel('Strike (K)')
    plt.ylabel('Price')
    plt.show()
