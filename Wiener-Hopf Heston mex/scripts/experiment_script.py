# does the overall numerical testing

from scripts.option_pricing_routine import evaluate_option_by_wh
from scripts.get_answer_for_prices import make_premialike_answers_for_folder
from scripts.method_to_mc_comparison import make_price_to_mc_comparison_for_folder
from shutil import copyfile
T = 1
H_original = 90  # limit
K_original = 100.0  # strike
r_premia = 10  # annual interest rate

# Heston model parameters
V0 = 0.1  # initial volatility
kappa = 2.0  # heston parameter, mean reversion
theta = 0.2  # heston parameter, long-run variance
sigma = omega = 0.2  # heston parameter, volatility of variance. Omega is used in variance tree, sigma - everywhere else
rho = 0.5  # heston parameter #correlation

# method parameters
N = 100  # number_of_time_steps
M = 2**11  # number of points in price grid
L = 3  # scaling coefficient


def make_experiment(T, H_original, K_original, r_premia, V0, kappa, theta, sigma, rho, N, M, L):
    evaluate_option_by_wh(T, H_original, K_original, r_premia, V0, kappa, theta, sigma, rho, N, M, L)

    filename = ("T=" + str(T) +
                "H=" + str(H_original) +
                "K=" + str(K_original) +
                "r=" + str(r_premia) +
                "V0=" + str(V0) +
                "kappa=" + str(kappa) +
                "theta=" + str(theta) +
                "sigma=" + str(sigma) +
                "rho=" + str(rho) +
                "N=" + str(N) +
                "M=" + str(M) +
                "L=" + str(L))

    experiment_result_file = open("../output/experiment/" + filename + ".csv", "w")
    routine_answer = open("../output/routine/answer_cumul.csv", "r")

    experiment_result_file.write("option parameters: \n")
    experiment_result_file.write("T=" + str(T) + '\n')
    experiment_result_file.write("H=" + str(H_original) + '\n')
    experiment_result_file.write("K=" + str(K_original) + '\n')
    experiment_result_file.write("r_premia=" + str(r_premia) + '\n')

    experiment_result_file.write("Heston model parameters: \n")
    experiment_result_file.write("V0=" + str(V0) + '\n')
    experiment_result_file.write("kappa=" + str(kappa) + '\n')
    experiment_result_file.write("theta=" + str(theta) + '\n')
    experiment_result_file.write("sigma=" + str(sigma) + '\n')
    experiment_result_file.write("rho=" + str(rho) + '\n')

    experiment_result_file.write("Method parameters: \n")
    experiment_result_file.write("N=" + str(N) + '\n')
    experiment_result_file.write("M=" + str(M) + '\n')
    experiment_result_file.write("L=" + str(L) + '\n')

    for line in routine_answer:
        experiment_result_file.write(line)

    experiment_result_file.close()
    routine_answer.close()
    copyfile("../output/figure.png", "../output/experiment/" + filename + ".png")

# for L in [2, 2.5, 5]:
#        for V0 in [0.1, 0.3, 0.5]:
#                    try:
#                         make_experiment(T, H_original, K_original, r_premia, V0, kappa, theta, sigma, rho, N, M, L)
#                    except:
#                        pass
make_experiment(T, H_original, K_original, r_premia, V0, kappa, theta, sigma, rho, N, M, L)
make_premialike_answers_for_folder("../output/experiment/", iterations=21, start_price=80, end_price=280)
make_price_to_mc_comparison_for_folder("../output/experiment_premia_prices/")
