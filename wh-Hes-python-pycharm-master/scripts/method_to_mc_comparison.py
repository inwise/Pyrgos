from scripts.heston_mc import calculate_heston_mc_price
from scripts.get_answer_for_prices import make_premialike_answers, read_paramset
import os

def make_price_to_mc_comparison(file_with_prices_and_parameters):
    paramset = read_paramset(file_with_prices_and_parameters)
    if not paramset.keys():
        print("cannot obtain parameters from the input file. Are there any? name = " + file_with_prices_and_parameters)
    else:
        mc_prices = []
        source_file = open(file_with_prices_and_parameters, 'r')
        for line in source_file:
            if line[0].isdigit():
                s0 = float(line.split(',')[0])
                heston_mc_price = calculate_heston_mc_price(paramset['T'], s0, paramset['H'], paramset['K'],
                                           paramset['r_premia'],
                                           paramset['V0'], paramset['kappa'], paramset['theta'],
                                           paramset['sigma'], paramset['rho'])
                mc_prices.append(heston_mc_price)
        source_file.close()
        return mc_prices


def create_price_to_mc_comparison_file(file_with_prices_and_parameters):
    source_file_name = file_with_prices_and_parameters
    source_file = open(file_with_prices_and_parameters, 'r')
    # changing filename and directory
    dest_file_name = source_file_name.replace(".csv", "_mc.csv")
    dest_file_name = source_file_name.replace(dest_file_name.split('/')[-2], "experiment_premia_vs_mc")
    dest_file = open(dest_file_name, 'w')

    mc_prices = make_price_to_mc_comparison(file_with_prices_and_parameters)

    i = 0
    for line in source_file:
        if not line[0].isdigit():
            dest_file.write(line)
        else:
            line_with_mc = line.split("\n")[0] + "," + str(mc_prices[i]) + "\n"
            dest_file.write(line_with_mc)
            i += 1


def make_price_to_mc_comparison_for_folder(experiment_folder):
    for source_file_name in os.listdir(experiment_folder):
        resulting_source_filename = experiment_folder + source_file_name
        if resulting_source_filename.endswith(".csv"):
            try:
                create_price_to_mc_comparison_file(resulting_source_filename)
            except:
                print("cannot make monte-carlo approximation for" + resulting_source_filename)

if __name__ == '__main__':
#    make_price_to_mc_comparison_for_folder("../output/experiment_premia_prices/")
    make_price_to_mc_comparison("/home/basil/PycharmProjects/Project_Wiener_Hopf_OP/output/experiment/"
                                "T=1H=90K=100.0r=10V0=0.01kappa=2.0theta=0.01sigma=0.2rho=0.5N=100M=512L=1.5.csv")
