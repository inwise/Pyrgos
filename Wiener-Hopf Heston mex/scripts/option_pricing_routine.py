import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import pandas as pd
from scripts.volatility_tree import build_volatility_tree
from scripts.profiler import profiler
i = complex(0, 1)

# option parameters
T = 1
H_original = 90  # limit
K_original = 100.0  # strike
r_premia = 10  # annual interest rate

# Bates model parameters
V0 = 0.01  # initial volatility
kappa = 2  # heston parameter, mean reversion
theta = 0.01  # heston parameter, long-run variance
sigma = omega = 0.2  # heston parameter, volatility of variance.
# Omega is used in variance tree, sigma - everywhere else
rho = 0.5  # heston parameter #correlation

# method parameters
N = 10  # number_of_time_steps
M = 2**15  # number of points in price grid
dx = 1e-3
omega_plus = 1
omega_minus = -1

r = np.log(r_premia/100 + 1)
omega = sigma

# time-space domain construction
x_space = np.linspace(-M * dx / 2, M * dx / 2, num=M, endpoint=False)
u_space = np.linspace(-pi / dx, pi / dx, num=M, endpoint=False)
du = u_space[1] - u_space[0]

first_step_of_return = np.array([elem + V0*rho/sigma for elem in x_space])
original_prices_array = H_original * np.exp(first_step_of_return)

delta_t = T/N

# making volatilily tree
markov_chain = build_volatility_tree(T, V0, kappa, theta, omega, N)
V = markov_chain[0]
pu_f = markov_chain[1]
pd_f = markov_chain[2]
f_up = markov_chain[3]
f_down = markov_chain[4]

rho_hat = np.sqrt(1 - rho**2)
q = 1.0/delta_t + r
factor = (q*delta_t)**(-1)


def G(S, K):
    """the payoff function of put option. Nothing to do with barrier"""
    return max(K-S, 0)

F_n_plus_1 = np.zeros((len(x_space), len(V[N])), dtype=complex)
F_n = np.zeros((len(x_space), len(V[N])), dtype=complex)
for j in range(len(x_space)):
    for k in range(len(V[N])):
        F_n_plus_1[j, k] = np.array(G(H_original * np.exp(x_space[j]), K_original))

# the global cycle starts here. It iterates over the volatility tree we just constructed, and goes backwards in time
# starting from n-1 position
# print("Main cycle entered")

# when the variance is less than that, is is reasonable to assume it to be zero, which leads to simpler calculations
treshold = 1e-6
discount_factor = np.exp(r*delta_t)


def psi(xi, gamma=0, sigma=sigma):
    return (sigma**2/2) * np.power(xi, 2) - 1j*gamma*xi


def make_rad_fft(f_x):

    sign_change_k = np.array([(-1)**k for k in range(0, M)])
    sign_change_l = np.array([(-1)**l for l in range(0, M)])
    # учитываем порядок хранения
    sign_change_l = np.fft.fftshift(sign_change_l)

    f = sign_change_k * f_x
    f_hat = dx * sign_change_l * np.fft.fft(f)

    # избегаем особенностей хранения результатов fft, нам они не нужны.
    return f_hat


def make_rad_ifft(f_hat_xi):

    M = len(f_hat_xi)

    sign_change_k = np.array([(-1)**k for k in range(0, M)])
    sign_change_l = np.array([(-1)**l for l in range(0, M)])

    f = (1/dx) * sign_change_k * np.fft.ifft(sign_change_l * f_hat_xi)
    return f


def make_phi_minus(gamma=0, sigma=sigma):

    def integrand_minus(upsilon_array):
        """
        принимает и возвращает массив длиной в степень двойки, исходя из логики дальнейшего использования
        """
        value = np.log(1 + psi(upsilon_array + 1j * omega_plus, gamma=gamma, sigma=sigma) / q) / (upsilon_array + 1j * omega_plus) ** 2
        return value

    def F_minus_capital():
        m_indicator = np.where(x_space >= 0, 1, 0)
        trimmed_x_space = m_indicator * x_space  # чтобы при "сильно отрицательных" x не росла экспонента
        integral = make_rad_ifft(integrand_minus(u_space))
        exponent = np.exp(-trimmed_x_space * omega_plus)
        return m_indicator * exponent * integral

    fm = F_minus_capital()
    F_m_hat = make_rad_fft(fm)

    def make_phi_minus_array(xi_array):
        first_term = - 1j * xi_array * (fm[M // 2])
        second_term = - xi_array * xi_array * F_m_hat
        return np.exp(first_term + second_term)

    mex_symbol_minus = make_phi_minus_array(u_space)
    return mex_symbol_minus


def make_phi_plus(gamma=0, sigma=sigma):

    def integrand_plus(upsilon_array):
        """
        принимает и возвращает массив длиной в степень двойки, исходя из логики дальнейшего использования
        """
        value = np.log(1 + psi(upsilon_array + 1j * omega_minus, gamma=gamma, sigma=sigma) / q) / (upsilon_array + 1j * omega_minus) ** 2
        return value

    def F_plus_capital():
        p_indicator = np.where(x_space <= 0, 1, 0)
        trimmed_x_space = p_indicator * x_space  # чтобы при "сильно отрицательных" x не росла экспонента
        integral = make_rad_ifft(integrand_plus(u_space))
        exponent = np.exp(-trimmed_x_space * omega_plus)
        return p_indicator * exponent * integral

    fp = F_plus_capital()

    F_p_hat = make_rad_fft(fp)

    def make_phi_plus_array(xi_array):
        first_term = 1j * xi_array * fp[M // 2]
        second_term = - xi_array * xi_array * F_p_hat
        return np.exp(first_term + second_term)

    mex_symbol_plus = make_phi_plus_array(u_space)
    return mex_symbol_plus

for n in range(len(V[N]) - 2, -1, -1):
    print(str(n) + " of " + str(len(V[N]) - 2))
    with profiler():
        for k in range(n+1):
            # to calculate the binomial expectation one should use Antonino's matrices f_up and f_down
            # the meaning of the containing integers are as follows - after (n,k) you will be in
            # either (n+1, k + f_up) or (n+1, k - f_down). We use k_u and k_d shorthands, respectively
            k_u = k + int(f_up[n][k])
            k_d = k + int(f_down[n][k])

            # initial condition of a step
            f_n_plus_1_k_u = np.array([F_n_plus_1[j][k_u] for j in range(len(x_space))])
            f_n_plus_1_k_d = np.array([F_n_plus_1[j][k_d] for j in range(len(x_space))])

            H_N_k = - (rho / sigma) * V[n, k]  # modified barrier
            local_domain = np.array([x_space[j] + H_N_k for j in range(len(x_space))])

            if V[n, k] >= treshold:
                # set up variance-dependent parameters for a given step
                sigma_local = rho_hat * np.sqrt(V[n, k])
                gamma = r - 0.5 * V[n, k] - rho / sigma * kappa * (theta - V[n, k])  # also local

                phi_plus_array = make_phi_minus(gamma=gamma, sigma=sigma_local)

                phi_minus_array = make_phi_plus(gamma=gamma, sigma=sigma_local)

                indicator = np.where(local_domain >= H_N_k, 1, 0)

                # factorization calculation
                f_n_k_u = factor * \
                          make_rad_ifft(phi_minus_array *
                                            make_rad_fft(
                                                indicator *
                                                    make_rad_ifft(phi_plus_array * make_rad_fft(f_n_plus_1_k_u))))

                f_n_k_d = factor * \
                          make_rad_ifft(phi_minus_array *
                                            make_rad_fft(
                                                indicator *
                                                    make_rad_ifft(phi_plus_array * make_rad_fft(f_n_plus_1_k_d))))
            elif V[n, k] < treshold:
                f_n_plus_1_k_u = [F_n_plus_1[j][k_u] for j in range(len(x_space))]
                f_n_k_u = discount_factor * f_n_plus_1_k_u

                f_n_plus_1_k_d = [F_n_plus_1[j][k_d] for j in range(len(x_space))]
                f_n_k_d = discount_factor * f_n_plus_1_k_d

            f_n_k = f_n_k_u * pu_f[n, k] + f_n_k_d * pd_f[n, k]

            for j in range(len(f_n_k)):
                # here we try some cutdown magic. The procedure without it returns great bubbles to the right
                # from the strike. And the more L the greater this bubble grows.
                # what we are going to do there is to try to cut off all the values on prices greater than, say,
                # 4 times bigger then the strike
                # we use S>4K and, therefore, y > ln(4K/H) + (pho/sigma)*V inequality to do this
                if local_domain[j] < np.log(3.5*K_original/H_original + (rho/sigma) * V[n][k]):
                    F_n[j][k] = f_n_k[j]
                else:
                    F_n[j][k] = complex(0)
            # plt.plot(original_prices_array, f_n_plus_1_k_u)
            # plt.show()


# for j in range(len(y)):
#    tree_to_csv_file(y[j], "../output/routine/price_slices/Y" + str(original_prices_array[j]) + ".csv")

# for j in range(len(F)):
#    tree_to_csv_file(F[j], "../output/routine/answers/F" + str(original_prices_array[j]) + ".csv")

answer_total = open("../output/routine/answer_cumul.csv", "w")

answers_list = np.array([F_n[j][0] for j in range(len(x_space))])
for elem in list(zip(original_prices_array, answers_list)):
    answer_total.write(str(elem[0]) + ',')
    answer_total.write(str(elem[1].real) + ',')
   # answer_total.write(str(elem[1].imag) + ',')
    answer_total.write('\n')
# for j in range(len(F)):
#    tree_to_csv_file(F[j], "../output/routine/answers/F" + str(original_prices_array[j]) + ".csv")

plt.plot(original_prices_array[(original_prices_array>75) & (original_prices_array<200)],
         answers_list[(original_prices_array>75) & (original_prices_array<200)])
plt.savefig("../output/figure.png")
plt.show()
plt.close()