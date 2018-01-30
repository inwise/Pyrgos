from math import log, sqrt

from numpy import exp, fft, array, pi, zeros
from scripts.simple_logging_funcs import tree_to_csv_file, array_to_csv_file
from scripts.useful_functions import G, indicator
import matplotlib.pyplot as plt

from scripts.volatility_tree import build_volatility_tree, tree_to_csv_file
i = complex(0, 1)

if __name__ == '__main__':

    # option parameters
    T = 1
    H_original = 90  # limit
    K_original = 100.0  # strike
    r_premia = 10  # annual interest rate

    # Heston model parameters
    V0 = 0.1  # initial volatility
    kappa = 2.0  # heston parameter, mean reversion
    theta = 0.1  # heston parameter, long-run variance
    sigma = omega = 0.2  # heston parameter, volatility of variance.
    # Omega is used in variance tree, sigma - everywhere else
    rho = 0.5  # heston parameter #correlation

    # method parameters
    N = 100  # number_of_time_steps
    M = 2**9  # number of points in price grid
    L = 3  # scaling coefficient


def evaluate_option_by_wh(T, H_original, K_original, r_premia, V0, kappa, theta, sigma, rho, N, M, L):
    if 2 * kappa * theta < sigma**2:
        print("Warning, Novikov condition is not satisfied, the volatility values could be negative")
    r = log(r_premia/100 + 1)
    omega = sigma

    # time-space domain construction
    x_min = L * log(0.5)
    x_max = L * log(2.0)
    d = (x_max - x_min) / float(M)  # discretization step
    x_space = []  # prices array
    for p_elem_index in range(M):
        x_space.append(x_min + p_elem_index * d)
#   array_to_csv_file(x_space, "../output/routine/price_line_log.csv")

    first_step_of_return = [elem + V0*rho/sigma for elem in x_space]
    original_prices_array = H_original * exp(first_step_of_return)
#   array_to_csv_file(original_prices_array, "../output/routine/price_line_original.csv")
#   time discretization

    delta_t = T/N

    # making volatilily tree
    markov_chain = build_volatility_tree(T, V0, kappa, theta, omega, N)
    V = markov_chain[0]
    pu_f = markov_chain[1]
    pd_f = markov_chain[2]
    f_up = markov_chain[3]
    f_down = markov_chain[4]

    # tree_to_csv_file(V, "../output/routine/variance_tree.csv")
    # tree_to_csv_file(pu_f, "../output/routine/up_jumps_probabilities_on_variance_tree.csv")
    # tree_to_csv_file(pd_f, "../output/routine/down_jumps_probabilities_on_variance_tree.csv")
    # tree_to_csv_file(f_up, "../output/routine/ku.csv")
    # tree_to_csv_file(f_down, "../output/routine/kd.csv")

    # we're now working on volatility tree scope.
    # here we are going to construct a substitution for S - original prices array
    # the substitution is unique for each node of the volatility tree.

    # y_j (n,k) = x_space_j - rho/sigma * V (n,k)

    # as soon as we cannot properly print out the 3-d array, we store the info on whole time-vol-space system
    # as a collection of files

    # print("total price tree construction started")

    xi_space = fft.fftfreq(M, d=d)
    rho_hat = sqrt(1 - rho**2)
    q = 1.0/delta_t + r
    factor = (q*delta_t)**(-1)
    # initially F stores only payout function, to be later filled with computational results for conditional expectation
    # on the Markov chain vertices. F[0,0] should store the final answer
    F_n_plus_1 = zeros((len(x_space), len(V[N])), dtype=complex)
    F_n = zeros((len(x_space), len(V[N])), dtype=complex)
    for j in range(len(x_space)):
        for k in range(len(V[N])):
            F_n_plus_1[j, k] = G(H_original * exp(x_space[j]), K_original)

    # the global cycle starts here. It iterates over the volatility tree we just constructed, and goes backwards in time
    # starting from n-1 position
    # print("Main cycle entered")

    # when the variance is less than that, is is reasonable to assume it to be zero, which leads to simpler calculations
    treshold = 1e-6
    discount_factor = exp(r*delta_t)

    for n in range(len(V[N]) - 2, -1, -1):
        print(str(n) + " of " + str(len(V[N]) - 2))
        for k in range(n+1):
            # to calculate the binomial expectation one should use Antonino's matrices f_up and f_down
            # the meaning of the containing integers are as follows - after (n,k) you will be in
            # either (n+1, k + f_up) or (n+1, k - f_down). We use k_u and k_d shorthands, respectively
            k_u = k + int(f_up[n][k])
            k_d = k + int(f_down[n][k])

            # initial condition of a step
            f_n_plus_1_k_u = array([F_n_plus_1[j][k_u] for j in range(len(x_space))])
            f_n_plus_1_k_d = array([F_n_plus_1[j][k_d] for j in range(len(x_space))])

            H_N_k = - (rho / sigma) * V[n, k]  # modified barrier
            local_domain = array([x_space[j] + H_N_k for j in range(len(x_space))])

            if V[n, k] >= treshold:
                # set up variance-dependent parameters for a given step
                sigma_local = rho_hat * sqrt(V[n, k])
                gamma = r - 0.5 * V[n, k] - rho/sigma * kappa * (theta - V[n, k]) # also local

                # beta_plus and beta_minus
                beta_minus = - (gamma + sqrt(gamma**2 + 2*sigma_local**2 * q))/sigma_local**2
                beta_plus = - (gamma - sqrt(gamma**2 + 2*sigma_local**2 * q))/sigma_local**2

                # factor functions
                phi_plus_array = array([beta_plus/(beta_plus - i*2*pi*xi) for xi in xi_space])
                phi_minus_array = array([-beta_minus/(-beta_minus + i*2*pi*xi) for xi in xi_space])

                # factorization calculation
                f_n_k_u = factor * \
                          fft.ifft(phi_minus_array *
                                            fft.fft(
                                                indicator(
                                                    fft.ifft(phi_plus_array * fft.fft(f_n_plus_1_k_u)),
                                                    local_domain, H_N_k)))

                f_n_k_d = factor * \
                          fft.ifft(phi_minus_array *
                                            fft.fft(
                                                indicator(
                                                    fft.ifft(phi_plus_array * fft.fft(f_n_plus_1_k_d)),
                                                    local_domain, H_N_k)))
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
                if local_domain[j] < log(3.5*K_original/H_original + (rho/sigma) * V[n][k]):
                    F_n[j][k] = f_n_k[j]
                else:
                    F_n[j][k] = complex(0)
        F_n_plus_1 = F_n[:, :-1]



    # for j in range(len(y)):
    #    tree_to_csv_file(y[j], "../output/routine/price_slices/Y" + str(original_prices_array[j]) + ".csv")

    # for j in range(len(F)):
    #    tree_to_csv_file(F[j], "../output/routine/answers/F" + str(original_prices_array[j]) + ".csv")

    answer_total = open("../output/routine/answer_cumul.csv", "w")

    answers_list = [F_n[j][0] for j in range(len(x_space))]
    for elem in list(zip(original_prices_array, answers_list)):
        answer_total.write(str(elem[0]) + ',')
        answer_total.write(str(elem[1].real) + ',')
       # answer_total.write(str(elem[1].imag) + ',')
        answer_total.write('\n')
    # for j in range(len(F)):
    #    tree_to_csv_file(F[j], "../output/routine/answers/F" + str(original_prices_array[j]) + ".csv")
    plt.plot(original_prices_array, answers_list)
    plt.savefig("../output/figure.png")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    evaluate_option_by_wh(T, H_original, K_original, r_premia, V0, kappa, theta, sigma, rho, N, M, L)
