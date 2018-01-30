from scripts.useful_functions import G
from scipy import stats
from numpy import sqrt, log, exp
from numpy.random import standard_normal
"""We will use Zanette's substitution there """
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
    r = log(r_premia / 100.0 + 1.0)
    dt = float(T)/float(N)
    sqrt_dt = sqrt(dt)
    # trajectory started

    # initials
    V_t = V0
    Y_t = log(S0/H) - (rho/sigma) * V_t

    # random_values_for_V = stats.norm.rvs(size=N)
    # random_values_for_Y = stats.norm.rvs(size=N)
    rho_hat = sqrt(1 - rho**2)
    for j in range(N):
        # random walk for V
        random_value_for_V = stats.norm.rvs()
        dZ_V = random_value_for_V * sqrt_dt

        # random walk for S + correlation
        random_value_for_Y = stats.norm.rvs()
        dZ_Y = random_value_for_Y * sqrt_dt

        # equation for V
        dV_t = kappa * (theta - V_t) * dt + sigma * sqrt(V_t) * sqrt_dt * dZ_V
        V_t += dV_t
        V_t = max(0, V_t)
        # equation for S
        dY_t = (r - 0.5*V_t - (rho/sigma)*kappa*(theta-V_t)) * dt + rho_hat * sqrt(V_t) * dZ_Y
        Y_t += dY_t
        # check barrier crossing on each step
        if Y_t <= -rho/sigma*V_t:
            return 0
    return G(H*exp(Y_t + (rho/sigma) * V_t), K)


def calculate_heston_mc_price(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho, trajectories=10000):
    monte_carlo_price = 0.0
    for i in range(trajectories):
        monte_carlo_price += generate_heston_trajectory_return(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho)
    print("mc_prince_done")
    result = monte_carlo_price / float(trajectories)
    return result

if __name__ == '__main__':
    import time
    for S0 in range(91, 136, 5):
        start_time = time.clock()
        print(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho)
        mc_price = calculate_heston_mc_price(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho, trajectories=15000)
        end_time = time.clock() - start_time
        print('the price is %2f, computed in %2f seconds' % (mc_price, end_time))