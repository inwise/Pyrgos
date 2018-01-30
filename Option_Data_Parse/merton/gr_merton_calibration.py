from scipy.optimize import fmin
from scipy.stats import uniform
from scipy import optimize
import merton.gr_merton_pricing as merton
import merton.gr_char_exponents as chex
from merton.simple_datafeed import sample_dreamfile


def error(x, market_datas):
    """
    Чтобы применять калибровку или подбирать нечто,
    нужно использовать некоторую функцию ошибок.
    Используется средняя квадратическая ошибка.

    :param x: - значения параметров модели Мертона, в порядке sigma, lambda, delta, mu
    :param market_datas: - рыночные данные, содержащие S0, K, put/call option type, Option_Price,r ,T
    :return: значение ошибки
    """

    sigma, lambda_m, delta, mu = x  # the name lambda_m is to avoid occasional redefinition of the keyword
    result = 0.0
    for market_data in market_datas:
        s0, k, put_or_call, market_price, r, t = market_data
        if put_or_call == 'c' or put_or_call == 'C':
            merton_price = merton.call_price(s0, k, chex.merton_char_exp, (sigma, lambda_m, delta, mu), r, t)
        elif put_or_call == 'p' or put_or_call == 'P':
            merton_price = merton.put_price(s0, k, chex.merton_char_exp, (sigma, lambda_m, delta, mu), r, t)
        else:
            print('Unknown option type')
            return 0.0

        result += (merton_price - market_price) ** 2
    result /= len(market_datas)
    # print(sigma, lambda_m, delta, mu, 'sqr_error = ', result)
    return result


def make_initial_parameters_guess(market_datas, trials=2):
    """
    Подбор начального приближения для калибровки модели Мертона

    К сожалению, генетические алгоритмы показали себя весьма жадными до времени исполнения в этих задачах.
    Полный перебор поверхностей занимал часы, калибровку для российского рынка в динамике за обозримое время
    сделать не получалось.

    Поэтому здесь мы используем наиболее примитивный вариант Монте-Карло, который занимает не так много времени:
    мы ищем хорошее начальное приближение посредством угадывания.

    Угадывание обеспечивается использованием равномерного распределения
    uniform.rvs(loc=-1, scale=2)
    в диапазоне (loc, loc+scale).

    Пока для всех параметров модели Мертона стоит (-1,1).

    :param market_datas: - рыночные данные, содержащие S0, K, Call_Price,r ,T
    :param trials: - количество точек для угадывания
    :return: [наименьшая из полученных ошибок, параметры, на которых она достигается]
    Параметры возвращаются в порядке sigma, lambda, delta, mu
    """
    all_parameters = {}

    for _ in range(trials):
        sigma = uniform.rvs(loc=0, scale=2)
        lambda_m = uniform.rvs(loc=-1, scale=2)
        delta = uniform.rvs(loc=-1, scale=2)
        mu = uniform.rvs(loc=-1, scale=2)
        x = (sigma, lambda_m, delta, mu)
        all_parameters[error(x, market_datas)] = x
    errors = list(all_parameters.keys())
    errors.sort(key=lambda er: float(er))
    best_error = errors[0]
    best_parameters = all_parameters[best_error]
    return best_error, best_parameters


def calibrate(init_val, market_datas, max_iterations=40):
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
    :return: (корень из среднеквадратической ошибки, набор параметров, на которых она достигается)
    """
    # opt = optimize.minimize(error, init_val, args=(market_datas,),
    #                         bounds=((0, None), (0, None,), (0, None), (None, None),),
    #                         method='l-bfgs-b', options={'maxiter': max_iterations, 'disp': 1, 'ftol': 10000})
    opt = fmin(error, init_val, args=(market_datas,), maxiter=max_iterations, disp=0, ftol=10000)
    return error(opt, market_datas)**0.5, opt

if __name__ == '__main__':
    import time
    time_start = time.clock()
    header, market_datas = sample_dreamfile()
    # Initialize vol=sigma, lambda, delta, mu
    header, market_datas = sample_dreamfile()
    guesses = 1
    init_error, init_val = make_initial_parameters_guess(market_datas, trials=guesses)
    print('initials are', init_error, init_val)
    time_elapsed_estimating = time.clock() - time_start
    print('time elapsed %2f sec, guesses made: %d' % (time_elapsed_estimating, guesses))
    # calibration of parameters
    time_start = time.clock()
    (sigma, lambda_m, delta, mu) = calibrate(init_val, market_datas, max_iterations=3)[1]
    print('optimals are: sigma = %2f, lambda = %2f, delta = %2f, mu = %2f' % (sigma, lambda_m, delta, mu))
    time_elapsed_calibration = time.clock() - time_start
    print('error in optimals is %2f' % error((sigma, lambda_m, delta, mu), market_datas)**0.5)
    print('time for calibration %2f sec' % time_elapsed_calibration)