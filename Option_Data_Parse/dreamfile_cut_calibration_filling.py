import pandas as pd
import heston.gh_heston_calibration as heston
import merton.gr_merton_calibration as merton
import time

start_time = time.clock()

content = pd.read_csv('dreamFile_cut.csv', delimiter=';')
strikes = [175000, 180000, 185000, 190000, 195000]
strike_colnames = ['Strike_' + str(strike) + '_AVG_Daily_Price' for strike in strikes]
needed_colnames = ['BA_price', 'Put/Call', 'Expiration']
needed_colnames.extend(strike_colnames)

content_for_calculation = content[needed_colnames]

calibration_info = pd.DataFrame()

for i in range(len(content_for_calculation)):  # 5 образцов данных представлено
    print('calibrating ', i, ' of ', (len(content_for_calculation)))
    options_datas_for_period = []
    content = content_for_calculation.iloc[i]
    for strike in strikes:
        s0 = content['BA_price']
        K = strike
        put_or_call = content['Put/Call']
        option_price = content['Strike_' + str(strike) + '_AVG_Daily_Price']
        r = 0.0  # наш актив - опционы на фьючерс, здесь можно не учитывать эти эффекты.
        T = content['Expiration']
        options_datas_for_period.append([s0, K, put_or_call, option_price, r, T])
    # калибруем модели
    heston_guess = heston.make_initial_parameters_guess(options_datas_for_period, trials=1)[1]
    heston_calibration_results = heston.calibrate(heston_guess, options_datas_for_period, max_iterations=1)
    heston_parameters = heston_calibration_results[1]
    heston_error = heston_calibration_results[0]
    string_summary_heston = pd.DataFrame({'h-kappa': [heston_parameters[0]],
                                          'h-theta': [heston_parameters[1]],
                                          'h-sigma': [heston_parameters[2]],
                                          'h-rho': [heston_parameters[3]],
                                          'h-v0': [heston_parameters[4]],
                                          'h-error': [heston_error]})

    merton_guess = merton.make_initial_parameters_guess(options_datas_for_period, trials=1)[1]
    merton_calibration_results = merton.calibrate(merton_guess, options_datas_for_period, max_iterations=1)
    merton_parameters = merton_calibration_results[1]
    merton_error = merton_calibration_results[0]
    # формируем структуру данных с результатами калибровки:
    string_summary_merton = pd.DataFrame({'m-sigma': [merton_parameters[0]],
                                          'm-lambda': [merton_parameters[1]],
                                          'm-delta': [merton_parameters[2]],
                                          'm-mu': [merton_parameters[3]],
                                          'm-error': [merton_error]})

    summary_row = string_summary_heston.join(string_summary_merton)
    calibration_info = calibration_info.append(summary_row, ignore_index=True)
    output = content_for_calculation.join(calibration_info)
    output.to_csv('calibration_cut_output.csv', sep=';')

end_time = time.clock() - start_time

print('time elapsed %2f' % end_time)
