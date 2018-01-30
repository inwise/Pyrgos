"""
Здесь считаем факторы как в мексиканской статье.
"""
import numpy as np
from numpy import exp, log
import scripts.rad_ffts as ffts


def psi_bs(xi, sigma, gamma):
    return (sigma ** 2 / 2) * xi * xi - 1j * gamma * xi


def psi_merton(xi, sigma, gamma, pois_lambda, pois_delta, pois_mu):
    jump_component = pois_lambda * (1 - exp(- 0.5 * (pois_delta**2 * xi**2) + 1j * pois_mu * xi))
    return (sigma ** 2 / 2) * xi * xi - 1j * gamma * xi + jump_component


def make_phi_minus(M, dx, omega_plus, gamma, sigma, q, pois_lambda, pois_delta, pois_mu):
    x_space = ffts.make_fft_spaces(M, dx)[0]
    u_space = ffts.make_fft_spaces(M, dx)[1]

    def integrand_minus_bates(upsilon_array, gamma, sigma, pois_lambda, pois_delta, pois_mu):
        """
        принимает и возвращает массив длиной в степень двойки, исходя из логики дальнейшего использования
        """
        return np.array([log(1 + psi_merton(upsilon + 1j * omega_plus, gamma, sigma, pois_lambda, pois_delta, pois_mu) \
                             / q) / (upsilon + 1j * omega_plus) ** 2 for upsilon in upsilon_array])

    def F_minus_capital():
        indicator = np.where(x_space >= 0, 1, 0)
        trimmed_x_space = indicator * x_space  # чтобы при "сильно отрицательных" x не росла экспонента
        integral = ffts.make_rad_ifft(integrand_minus(u_space, gamma, sigma, pois_lambda, pois_delta, pois_mu), dx)
        exponent = exp(-trimmed_x_space * omega_plus)
        return indicator * exponent * integral

    fm = F_minus_capital()
    F_m_hat = ffts.make_rad_fft(fm, dx)

    def make_phi_minus_array(xi_array):
        first_term = - 1j * xi_array * (fm[M // 2])
        second_term = - xi_array * xi_array * F_m_hat
        return exp(first_term + second_term)

    mex_symbol_minus = make_phi_minus_array(u_space)
    return mex_symbol_minus


def make_phi_plus(M, dx, omega_minus, gamma, sigma, q, pois_lambda, pois_delta, pois_mu):
    x_space = ffts.make_fft_spaces(M, dx)[0]
    u_space = ffts.make_fft_spaces(M, dx)[1]

    def integrand_plus(upsilon_array, gamma, sigma, pois_lambda, pois_delta, pois_mu):
        return np.array([log(1 + psi_merton(upsilon + 1j * omega_minus, gamma, sigma, pois_lambda, pois_delta, pois_mu)\
                             / q) / (upsilon + 1j * omega_minus) ** 2 for upsilon in upsilon_array])

    def F_plus_capital():
        indicator = np.where(x_space <= 0, 1, 0)
        trimmed_x_space = indicator * x_space  # чтобы при "сильно положительных" x не росла экспонента
        integral = ffts.make_rad_ifft(integrand_plus(u_space, gamma, sigma, pois_lambda, pois_delta, pois_mu), dx)
        exponent = exp(-trimmed_x_space * omega_minus)
        return indicator * exponent * integral

    fp = F_plus_capital()

    F_p_hat = ffts.make_rad_fft(fp, dx)

    def make_phi_plus_array(xi_array):
        first_term = 1j * xi_array * fp[M // 2]
        second_term = - xi_array * xi_array * F_p_hat
        return exp(first_term + second_term)

    mex_symbol_plus = make_phi_plus_array(u_space)
    return mex_symbol_plus

if __name__ == '__main__':
    T = 1
    H_original = 90.0  # limit
    K_original = 100.0  # strike
    r_premia = 10  # annual interest rate
    r = log(r_premia / 100 + 1)
    V0 = 0.316227766

    sigma = V0
    gamma = r - 0.5 * sigma ** 2  # Black-Scholes parameter, from a no-arbitrage condition

    N = 5  # количество шагов по времени
    delta_t = T / N
    print('Шаг по времени равен', delta_t)

    q = 1.0 / delta_t + r
    print('q = ', q)
    factor = (q * delta_t) ** (-1)
    print('Множитель каждого шага =', factor)

    omega_plus = 1
    omega_minus = -1

    poisson_lambda = 0.1
    poisson_delta = 0.1
    poisson_mu = 0.1


    M = 2**17
    dx = 1e-3
    phi_m = make_phi_minus(M, dx, omega_plus, gamma, sigma, q, poisson_lambda, poisson_delta, poisson_mu)
    phi_p = make_phi_plus(M, dx, omega_minus, gamma, sigma, q, poisson_lambda, poisson_delta, poisson_mu)

    import matplotlib.pyplot as plt
    # plt.plot(phi_m.real, 'g')
    # plt.show()
    # plt.plot(phi_m.imag, 'r')
    # plt.show()
    # plt.plot(phi_m.real, 'g')
    # plt.show()
    # plt.plot(phi_m.imag, 'r')
    # plt.show()

    phi_p_hat = ffts.make_rad_fft(phi_p, dx)
    plt.plot(phi_p_hat.real[2**16-4:2**16+200], 'g.')
    plt.show()


