import merton.gr_char_exponents as ch_ex
from numpy import log, exp, pi


def call_price(s, k, char_exp, p, r, t):
    # не очень ясно, по какой формуле
    n = 10000
    kl = log(k/s)
    sum = 0.0
    a = 1.5
    up = 30.0
    nu = up/n
    v = 0.0
    for i in range(n):
        sum += exp(-1j*v*kl)*ch_ex.psi(char_exp, p, v, a, r, t)*nu
        v += nu
    c = exp(-a*kl)/pi*sum*s
    return c.real


def put_price(s, k, char_exp, p, r, t):
    # через паритет
    c = call_price(s, k, char_exp, p, r, t)
    return c-s+k*exp(-r*t)

if __name__ == "__main__":
    import time

    s, t, vol, k = (132620.0, 0.2, 0.3, 100000.0)
    time_start = time.clock()
    # 32914.8336384 is correct, by original test. TODO: find some good calibration value to be sure
    print(call_price(s, k, ch_ex.merton_char_exp, (vol, 1.0, 0.2, 0.2), 0.0, t))
    time_elapsed = (time.clock() - time_start)
    print('time elapsed: %2f seconds' % time_elapsed)
