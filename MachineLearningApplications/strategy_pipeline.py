import pandas as pd
import position_maker
import strategy_metrics
import strategy_stats
import plotting
"""
В этом файле должны быть собраны методы из position_maker, strategy_metrics и strategy_stats.
Их довольно неудобно править по отдельности и намного лучше, если их получится запускать все вместе,
получая на выходе итоговый файл со статисткой о поведении стратегии.

Рисование тоже логично расположить здесь.
"""

if __name__ == '__main__':
    # Здесь нужно заменить названия файлов и директорий на свои
    my_data_folder = "/home/basil/Documents/findata/customs/"
    my_data_filename = "SBER_RSI"
    try:

        # 1 strategy markup
        df = pd.read_csv(my_data_folder + my_data_filename + ".csv")
        positions_col = position_maker.make_positions(df)
        positions_frame = pd.DataFrame({'<POSITION>': positions_col})
        df = df.join(positions_frame)

        # 2 strategy details
        validity_info = strategy_metrics.check_strategy_validity(df)
        if validity_info == 'ok':
            deals = strategy_metrics.collect_deals(df)
            portfolio_performance = []
            deal_results = [float(info[0]) for info in deals]
            deal_positions = [int(info[1]) for info in deals]
            deal_directions = [info[2] for info in deals]
            buy_and_hold_profit = df['<CLOSE>'].values[-1] - df['<CLOSE>'].values[0]
            deals_info_frame = pd.DataFrame({'<DEAL_RESULT>': deal_results, '<DEAL_DIRECTION>': deal_directions},
                                            index=[deal_positions])
            df = df.join(deals_info_frame)
            df['<PERFORMANCE>'] = df['<DEAL_RESULT>'].cumsum()
        else:
            raise UserWarning('Something is wrong with the strategy somehow. Check it')

        # 3 logging output and metrics
        buy_and_hold_profit = df['<CLOSE>'].values[-1] - df['<CLOSE>'].values[0]

        total_strategy_income = strategy_stats.calculate_total_profit(df)
        max_portfolio_value = strategy_stats.calculate_max_profit(df)
        deepest_drawdown = strategy_stats.calculate_deepest_drawdown(df)
        deals_count = strategy_stats.count_deals(df)
        long_deals_count = strategy_stats.count_long_deals(df)
        short_deals_count = strategy_stats.count_short_deals(df)
        good_deals = strategy_stats.count_good_deals(df)
        bad_deals = strategy_stats.count_bad_deals(df)
        average_profit = strategy_stats.calculate_average_profit(df)
        good_long_deals = strategy_stats.count_good_long_deals(df)
        bad_long_deals = strategy_stats.count_bad_long_deals(df)
        average_long_deal = strategy_stats.calculate_average_long_deal_profit(df)
        good_short_deals = strategy_stats.count_good_short_deals(df)
        bad_short_deals = strategy_stats.count_bad_short_deals(df)
        average_short_deal = strategy_stats.calculate_average_short_deal_profit(df)

        log_strings = [str('0. Buy_and_hold_profit: %.2f' % buy_and_hold_profit),
                       str('1. total_strategy_income %.2f' % total_strategy_income),
                       str('2. max_portfolio_value %.2f' % max_portfolio_value),
                       str('3. deepest_drawdown %.2f' % deepest_drawdown),
                       str('4. deals_count %d' % deals_count),
                       str('5. long_deals_count %d' % long_deals_count),
                       str('6. short_deals_count %d' % short_deals_count),
                       str('7. good_deals %d (%.2f%%)' % (good_deals, 100 * float(good_deals)/deals_count)),
                       str('8. bad_deals %d (%.2f%%)' % (bad_deals, 100 * float(bad_deals)/deals_count)),
                       str('9. average_profit %.2f' % average_profit),
                       str('10. good_long_deals %d (%.2f%%)' % (good_long_deals,
                                                                100 * float(good_long_deals)/long_deals_count)),
                       str('11. bad_long_deals %d (%.2f%%)' % (bad_long_deals,
                                                               100 * float(bad_long_deals)/long_deals_count)),
                       str('12. average_long_deal %.2f' % average_long_deal),
                       str('13. good_short_deals %d (%.2f%%)' % (good_short_deals,
                                                                 100 * float(good_short_deals)/short_deals_count)),
                       str('14. bad_short_deals %d (%.2f%%)' % (bad_short_deals,
                                                                100 * float(bad_short_deals)/short_deals_count)),
                       str('15. average_short_deal %.2f' % average_short_deal)
                       ]

        log_strings_rus = [str('0. "Купи-и-держи": %.2f' % buy_and_hold_profit),
                           str('1. Общая прибыль по стратегии: %.2f' % total_strategy_income),
                           str('2. Максимальная прибыль: %.2f' % max_portfolio_value),
                           str('3. Максимальная просадка: %.2f' % deepest_drawdown),
                           str('4. Общее число сделок: %d' % deals_count),
                           str('5. Число сделок на покупку: %d' % long_deals_count),
                           str('6. Число сделок на продажу: %d' % short_deals_count),
                           str('7. Прибыльных сделок: %d (%.2f%%)' % (good_deals, 100 * float(good_deals) / deals_count)),
                           str('8. Убыточных сделок: %d (%.2f%%)' % (bad_deals, 100 * float(bad_deals) / deals_count)),
                           str('9. Средняя прибыль за сделку: %.2f' % average_profit),
                           str('10. Прибыльных сделок на покупку: %d (%.2f%%)' % (good_long_deals,
                               100 * float(good_long_deals) / long_deals_count)),
                           str('11. Убыточных сделкок на покупку: %d (%.2f%%)' % (bad_long_deals,
                               100 * float(bad_long_deals) / long_deals_count)),
                           str('12. Средняя прибыль за сделку на покупку: %.2f' % average_long_deal),
                           str('13. Прибыльных сделок на продажу: %d (%.2f%%)' % (good_short_deals,
                               100 * float(good_short_deals) / short_deals_count)),
                           str('14. Убыточных сделкок на продажу: %d (%.2f%%)' % (bad_short_deals,
                               100 * float(bad_short_deals) / short_deals_count)),
                           str('15. Средняя прибыль за сделку на продажу: %.2f' % average_short_deal)
                           ]

        for string in log_strings:
            print(string)

        logfile = open(my_data_folder + my_data_filename + '_stats.txt', 'w')
        logfile.writelines([log_string + '\n' for log_string in log_strings])
        logfile = open(my_data_folder + my_data_filename + '_ru_stats.txt', 'w')
        logfile.writelines([log_string + '\n' for log_string in log_strings_rus])
        logfile.close()
        # plotting.draw_strategy_performance(df, my_data_folder + my_data_filename, label='greedy trendfollow - ' + df['<TICKER>'][0])
        plotting.draw_multiplot(df, my_data_folder + my_data_filename, label='greedy trendfollow')

    except UnicodeDecodeError:
        print("Cannot parse a file. Perhaps some wrong character in the file?")
    except OSError:
        print("No such file or cannot read/write. Make sure everything is ok about this.")
    # except KeyError:
    #     print("Error while parsing a file by pandas. "
    #           "Make sure the file is a consistent .csv and the delimiter is correct")
    except TypeError:
        print("The computation began. but there's some error in a file. Check the data inside")
    except UserWarning:
        print("Strategy error. Something is broken in a strategy algorithm or a file.")

    else:
        # print(df.head())
        try:
            df.to_csv(my_data_folder + my_data_filename + "_with_strategy.csv")
        except OSError:
            print("No such file or cannot read/write. Make sure everything is ok about this.")
