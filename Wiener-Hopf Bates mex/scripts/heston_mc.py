from scipy import stats
import numpy as np

if __name__ == '__main__':

    # option parameters
    T = 1
    S0 = 96.0
    H = 90  # limit
    K = 100.0  # strike
    r_premia = 10  # annual interest rate

    # Heston model parameters
    V0 = 0.01  # initial volatility
    kappa = 2.0  # heston parameter, mean reversion
    theta = 0.01  # heston parameter, long-run variance
    sigma = omega = 0.2  # heston parameter, volatility of variance.
    # Omega is used in variance tree, sigma - everywhere else
    rho = 0.5  # heston parameter #correlation


def generate_heston_trajectory_return(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho, N = 1000):
    """simulates Heston monte-carlo for Down-and-out put directly through equations"""
    r = np.log(r_premia / 100 + 1)
    dt = float(T)/float(N)
    sqrt_dt = np.sqrt(dt)
    # trajectory started

    # initials
    S_t = S0
    V_t = V0

    random_values_for_V = stats.norm.rvs(size=N)
    random_values_for_S_uncorrelated = stats.norm.rvs(size=N)
    for j in range(N):
        # random walk for V
        random_value_for_V = random_values_for_V[j]
        dZ_V = random_value_for_V * sqrt_dt

        # random walk for S + correlation
        random_value_for_S = rho * random_value_for_V + np.sqrt(1 - pow(rho, 2)) * random_values_for_S_uncorrelated[j]
        dZ_S = random_value_for_S * sqrt_dt

        # equation for V
        dV_t = kappa * (theta - V_t) * dt + sigma * np.sqrt(V_t) * sqrt_dt * dZ_V
        V_t += dV_t
        V_t = max(0,V_t)
        # equation for S
        dS_t = S_t * r * dt + S_t * np.sqrt(V_t) * dZ_S
        S_t += dS_t
        # check barrier crossing on each step
        if S_t <= H:
            return 0

    return max(0, K-S_t)


def calculate_heston_mc_price(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho, trajectories=10000):

    r = np.log(r_premia / 100 + 1)
    monte_carlo_price = 0.0
    mc_price_square = 0.0

    for i in range(trajectories):
        payoff = np.exp(-r*T) * generate_heston_trajectory_return(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho)
        monte_carlo_price += payoff
        mc_price_square += np.power(payoff, 2)
        if i % 500 == 0:
            print('trajectory', i, 'of', trajectories)

    variance_of_payoff = (mc_price_square - (monte_carlo_price*monte_carlo_price)/trajectories)/(trajectories-1)
    result = monte_carlo_price / float(trajectories)
    price_error = 1.96*np.sqrt(variance_of_payoff)/np.sqrt(trajectories)
    return result, price_error

if __name__ == '__main__':
    import time
    paths = 100000
    res_file = open('results_mc_' + str(paths) + '.txt', 'w')
    for S0 in range(91, 136, 5):
        start_time = time.clock()
        print(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho)
        mc_price, mc_error = calculate_heston_mc_price(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho,
                                                       trajectories=paths)
        print("mc_price_done for " + str(S0))
        print('S0:', str(S0), ';', 'price', str(mc_price), ';', 'price_error:', str(mc_error), file=res_file)
        end_time = time.clock() - start_time
        print('the price is %2f, computed in %2f seconds' % (mc_price, end_time))
        print('the price is %2f, computed in %2f seconds' % (mc_price, end_time), file=res_file)
    res_file.close()
