from numpy import exp


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


def merton_char_exp(u, p, r=0.0):
    # характеристическая экспонента для модели Мертона
    # p[0] - sigma>0, p[1] - lambda>0, p[2] - delta>0, p[3] - mu
    # Похоже, u = ksi, lj = complex(j)
    g0 = r-p[0]*p[0]/2.0+p[1]*(1-exp(p[2]*p[2]/2.0+p[3]))
    return p[0]*p[0]/2.0*u*u-1j*g0*u+p[1]*(1-exp(-p[2]*p[2]/2.0*u*u+1j*p[3]*u))

# TODO: find some good values to test all the functions from here
