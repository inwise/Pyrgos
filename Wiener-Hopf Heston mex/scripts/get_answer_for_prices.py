from numpy import linspace
from scipy import interpolate
import os

# at the matter of fact, the data is being split by ','
def semicolon_split(line):
    return line.split(',')


def get_digital_data_from_file(raw_prices_file_name):
    try:
        src_file = open(raw_prices_file_name, 'r')
        lines = []
        for line in src_file:
            if line[0].isdigit():
                lines.append(line)
        content = [semicolon_split(line) for line in lines]
        src_file.close()
        return content
    except:
        pass


def choose_price_info(price, content):
    differences = [abs(float(info_pair[0]) - price) for info_pair in content]
    nearest_price = min(differences)
    return content[differences.index(nearest_price)]


def choose_price_info_quadratic_interpolation(prices_list, content):
    derivative_prices = [float(info_pair[1]) for info_pair in content]
    script_ba_prices = [float(info_pair[0]) for info_pair in content]
    premia_ba_prices = [float(elem) for elem in prices_list]
    interpolated_prices_func = interpolate.interp1d(script_ba_prices, derivative_prices, kind='quadratic')
    interpolated_prices = [interpolated_prices_func(price) for price in premia_ba_prices]
    return list(zip(premia_ba_prices, interpolated_prices))


def choose_price_range_info(prices_list):
    output_list = []
    for price in prices_list:
        output_list.append(choose_price_info(price))
    return output_list


def make_premialike_answers(source_file_name, iterations=10, start_price=90, end_price=200):
    content = get_digital_data_from_file(source_file_name)

    # changing filename and directory
    destination_file_name = source_file_name.replace(".csv", "_premialike.csv")
    destination_file_name = destination_file_name.replace(destination_file_name.split('/')[-2], "experiment_premia_prices")
    dest_file = open(destination_file_name, 'w')

    # adding parameters info
    src_file = open(source_file_name, 'r')
    for line in src_file:
        if not line[0].isdigit():
            dest_file.write(line)
    prices_premialike = linspace(start_price, end_price, iterations)
    answer_premialike = choose_price_info_quadratic_interpolation(prices_premialike, content)
    for elem in answer_premialike:
        dest_file.write(str(elem[0]) + ',' + str(elem[1]) + ',' + '\n')
    dest_file.close()


def make_premialike_answers_for_folder(source_directory_name, iterations=10, start_price=90, end_price=200):
    try:
        for source_file_name in os.listdir(source_directory_name):
            resulting_source_filename = source_directory_name + source_file_name
            if resulting_source_filename.endswith(".csv"):
                make_premialike_answers(resulting_source_filename, iterations, start_price, end_price)
    except:
        pass


def read_paramset(raw_prices_file_name):
    parameter_holding_file = open(raw_prices_file_name, 'r')
    method_parameters = {}
    for i in range(15):
        line = parameter_holding_file.readline()
        # option parameters: T, H, K, r
        if line.startswith("T="):
            method_parameters["T"] = float(line.split('=')[1])
        elif line.startswith("H="):
            method_parameters["H"] = float(line.split('=')[1])
        elif line.startswith("K="):
            method_parameters["K"] = float(line.split('=')[1])
        elif line.startswith("r_premia="):
            method_parameters["r_premia"] = float(line.split('=')[1])
        # Heston model parameters V0, kappa, theta, sigma, rho
        elif line.startswith("V0="):
            method_parameters["V0"] = float(line.split('=')[1])
        elif line.startswith("kappa="):
            method_parameters["kappa"] = float(line.split('=')[1])
        elif line.startswith("theta="):
            method_parameters["theta"] = float(line.split('=')[1])
        elif line.startswith("sigma="):
            method_parameters["sigma"] = float(line.split('=')[1])
        elif line.startswith("rho="):
            method_parameters["rho"] = float(line.split('=')[1])
        # method parameters N, M, L
        elif line.startswith("N="):
            method_parameters["N"] = float(line.split('=')[1])
        elif line.startswith("M="):
            method_parameters["M"] = float(line.split('=')[1])
        elif line.startswith("L="):
            method_parameters["L"] = float(line.split('=')[1])
    return method_parameters

if __name__ == '__main__':
    raw_prices_file_name = "../output/experiment _old/" \
                           "T=1H=90K=100.0r =10V0=0.01kappa=2.0theta=0.01sigma0.2rho0.5N=100M=1024L=3.csv"
    make_premialike_answers(raw_prices_file_name, iterations=21, start_price=80, end_price=280)
