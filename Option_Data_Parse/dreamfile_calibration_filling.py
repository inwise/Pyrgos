import pandas as pd
import heston.gh_heston_calibration as heston
import merton.gr_merton_calibration as merton
import time

start_time = time.clock()

content = pd.read_csv('options_parse_vol.csv', delimiter=',')
strikes = [140000, 145000, 150000, 155000, 160000]

calibration_info = pd.DataFrame()

heston_error = 10000
merton_error = 10000

for i in range(len(content)):  # 5 образцов данных представлено
    start_iter_time = time.clock()
    print('calibrating ', i, ' of ', (len(content)))
    options_datas_for_period = []
    sample = content.iloc[i]
    for strike in strikes:
        s0 = sample['Ri']
        K = strike
        put_or_call = sample['type']
        option_price = sample['Strike_' + str(strike) + '_AVG_Daily_Price']
        r = 0.0  # наш актив - опционы на фьючерс, здесь можно не учитывать эти эффекты.
        T = sample['exp_time']
        options_datas_for_period.append([s0, K, put_or_call, option_price, r, T])
    # калибруем модели
    if i == 0:  # пробуем угадать на первом шаге
        heston_guess = heston.make_initial_parameters_guess(options_datas_for_period, trials=100)[1]
        heston_calibration_results = heston_guess
        heston_error = heston.error(heston_guess, options_datas_for_period)

    elif heston_error > 500:
        heston_guess = heston.make_initial_parameters_guess(options_datas_for_period, trials=50)[1]
    elif heston_error <= 500:  # уже откалибровались успешно однажды
        heston_guess = heston_parameters

    heston_calibration_results = heston.calibrate(heston_guess, options_datas_for_period, max_iterations=50)
    heston_parameters = heston_calibration_results[1]
    heston_error = heston_calibration_results[0]
    string_summary_heston = pd.DataFrame({'h-kappa': [heston_parameters[0]],
                                          'h-theta': [heston_parameters[1]],
                                          'h-sigma': [heston_parameters[2]],
                                          'h-rho': [heston_parameters[3]],
                                          'h-v0': [heston_parameters[4]],
                                          'h-error': [heston_error]})
    heston_time = time.clock() - start_iter_time

    merton_time = time.clock()
    if i == 0:  # пробуем угадать на первом шаге
        merton_guess = heston.make_initial_parameters_guess(options_datas_for_period, trials=100)[1]
        merton_calibration_results = heston_guess
        merton_error = heston.error(heston_guess, options_datas_for_period)

    if merton_error > 500:
        merton_guess = merton.make_initial_parameters_guess(options_datas_for_period, trials=50)[1]
    elif heston_error <= 500:  # уже откалибровались успешно однажды
        merton_guess = merton_parameters
    merton_calibration_results = merton.calibrate(merton_guess, options_datas_for_period, max_iterations=50)
    merton_parameters = merton_calibration_results[1]
    merton_error = merton_calibration_results[0]
    merton_time = time.clock() - merton_time
    # формируем структуру данных с результатами калибровки:
    string_summary_merton = pd.DataFrame({'m-sigma': [merton_parameters[0]],
                                          'm-lambda': [merton_parameters[1]],
                                          'm-delta': [merton_parameters[2]],
                                          'm-mu': [merton_parameters[3]],
                                          'm-error': [merton_error]})

    summary_row = string_summary_heston.join(string_summary_merton)
    calibration_info = calibration_info.append(summary_row, ignore_index=True)
    output = content.join(calibration_info)
    output.to_csv('calibration_big_output.csv', sep=',')
    time_for_iteration = time.clock() - start_iter_time
    print('Heston calibration time: %2f, Merton calibration time: %2f, For iteration total: %2f'
          % (heston_time, merton_time, time_for_iteration))
end_time = time.clock() - start_time
print('time elapsed %2f' % end_time)