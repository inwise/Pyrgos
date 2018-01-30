# -*- coding: utf-8 -*-
from math import * 
from scipy.stats import norm
from scipy.optimize import fmin_bfgs


# Black Sholes Function
def price(S, K, T, r, v, callPutFlag = 'c'):
    d1 = (log(S / K) + (r + 0.5 * v**2) * T) / (v * sqrt(T))
    d2 = d1 - v * sqrt(T)
    if (callPutFlag == 'c') or (callPutFlag == 'C'):
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# Calc Implied volatility
def implied_volatility(price_, S, K, T, r, callPutFlag = 'c'):
    Objective = lambda x: (price_ - price(S, K, T, r, x, callPutFlag))**2    
    return fmin_bfgs(Objective, 1, disp=False)[0]

# test
if __name__ == '__main__':
    import time
    time_start = time.clock()
    # correct : call 3.68
    print(price(49.0, 50.0, 1.0, 0.01, 0.2, 'C'))
    time_elapsed = time.clock() - time_start
    print('time elapsed: %2f seconds' % time_elapsed)

    time_start = time.clock()
    # correct : put 4.18
    print(price(49.0, 50.0, 1.0, 0.01, 0.2, 'P'))
    time_elapsed = time.clock() - time_start
    print('time elapsed: %2f seconds' % time_elapsed)

    time_start = time.clock()
    # correct : 0.2
    print(implied_volatility(3.68, 49.0, 50.0, 1.0, 0.01, 'C'))
    time_elapsed = time.clock() - time_start
    print('time elapsed: %2f seconds' % time_elapsed)

    time_start = time.clock()
    # correct : 0.2
    print(implied_volatility(4.18, 49.0, 50.0, 1.0, 0.01, 'P'))
    time_elapsed = time.clock() - time_start
    print('time elapsed: %2f seconds' % time_elapsed)
