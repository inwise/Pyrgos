# python 3

import pandas as pd
import numpy as np
"""

Здесь мы будем собирать вообще все метрики по стратегии, которые могут нас заинтересовать.
Файл, который нужен здесь на входе, содержит:

Информацию о дате (для графиков, если захотим): '<DATE>'
Информацию о ценах закрытия: '<CLOSE>'
Информацию о направлении сделок (типе позиции): '<DEAL_DIRECTION>'
Информацию об итогах каждой из сделок: '<DEAL_RESULT>'

Метрики, которые нас интересуют, таковы:
1. Итоговая прибыль
2. Максимальная прибыль
3. Максимальная просадка
4. Количество сделок
5. Прибыль от длинных позиций
6. Прибыль от коротких позиций
7. Число прибыльных сделок
8. Число убыточных сделок
9. Средняя прибыль за сделку
10. Число прибыльных длинных сделок
11. Число убыточных длинных сделок
12. Средняя прибыль за длинную сделку
13. Число прибыльных коротких сделок
14. Число убыточных коротких сделок
15. Средняя прибыль за короткую сделку

"""


def calculate_total_profit(df):
    """
    1. Считает итоговую прибыль

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - итог применения стратегии

    """
    return df.dropna()['<PERFORMANCE>'].values[-1]


def calculate_max_profit(df):
    """
    2. Считает максимальную полученную прибыль по стратегии за время торгов

    :param df: - датафрейм с колонкой '<PERFORMANCE>'
    :return: - максимальная прибыль

    """
    return max(df.dropna()['<PERFORMANCE>'].values)


def calculate_deepest_drawdown(df):
    """
    3. Считает максимальную просадку по стратегии за время торгов

    :param df: - датафрейм с колонкой '<PERFORMANCE>'
    :return: - максимальную просадку стратегии

    """
    return min(df.dropna()['<PERFORMANCE>'].values)


def count_deals(df):
    """
    4. Считает количество сделок

    :param df: - датафрейм с колонкой '<DEAL_DIRECTION>'
    :return: - общее количество сделок

    """
    return df.dropna()['<DEAL_DIRECTION>'].count()


def count_long_deals(df):
    """
    5. Считает количество сделок по длинной позиции

    :param df: - датафрейм с колонкой '<DEAL_DIRECTION>'
    :return: - общее количество сделок LONG

    """
    # http://stackoverflow.com/questions/27140860/count-occurrences-of-number-by-column-in-pandas-data-frame?rq=1

    return (df['<DEAL_DIRECTION>'] == 'LONG').sum()


def count_short_deals(df):
    """
    6. Считает количество сделок в короткой позиции

    :param df: - датафрейм с колонкой '<DEAL_DIRECTION>'
    :return: - общее количество сделок SHORT

    """
    # http://stackoverflow.com/questions/27140860/count-occurrences-of-number-by-column-in-pandas-data-frame?rq=1

    return (df['<DEAL_DIRECTION>'] == 'SHORT').sum()


def count_good_deals(df):
    """
    7. Считает число прибыльных сделок

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - число прибыльных сделок

    """

    # http://stackoverflow.com/questions/27140860/count-occurrences-of-number-by-column-in-pandas-data-frame?rq=1

    return (df['<DEAL_RESULT>'] > 0).sum()


def count_bad_deals(df):
    """
    8. Считает число убыточных сделок

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - число прибыльных сделок

    """

    # http://stackoverflow.com/questions/27140860/count-occurrences-of-number-by-column-in-pandas-data-frame?rq=1

    return (df['<DEAL_RESULT>'] < 0).sum()


def calculate_average_profit(df):
    """
    9. Считает среднюю прибыль за сделку

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - число прибыльных сделок

    """

    return df['<DEAL_RESULT>'].mean()


def count_good_long_deals(df):
    """
    10. Считает число прибыльных длинных сделок

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - число прибыльных сделок

    """

    return ((df['<DEAL_RESULT>'] > 0) & np.where(df['<DEAL_DIRECTION>'] == 'LONG', 1, 0)).sum()


def count_bad_long_deals(df):
    """
    11. Считает число убыточных длинных сделок

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - число прибыльных сделок

    """
    return ((df['<DEAL_RESULT>'] < 0) & np.where(df['<DEAL_DIRECTION>'] == 'LONG', 1, 0)).sum()


def calculate_average_long_deal_profit(df):
    """
    12. Считает среднюю прибыль длинную за сделку

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - средняя прибыль по длинной сделке

    """

    return ((df['<DEAL_RESULT>']) * np.where(df['<DEAL_DIRECTION>'] == 'LONG', 1, np.nan)).mean()


def count_good_short_deals(df):
    """
    13. Считает число прибыльных коротких сделок

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - число прибыльных сделок

    """
    return ((df['<DEAL_RESULT>'] > 0) & np.where(df['<DEAL_DIRECTION>'] == 'SHORT', 1, 0)).sum()


def count_bad_short_deals(df):
    """
    14. Считает число убыточных коротких сделок

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - число прибыльных сделок

    """

    return ((df['<DEAL_RESULT>'] < 0) & np.where(df['<DEAL_DIRECTION>'] == 'SHORT', 1, 0)).sum()


def calculate_average_short_deal_profit(df):
    """
    15. Считает среднюю прибыль за сделку

    :param df: - датафрейм с колонкой '<DEAL_RESULT>'
    :return: - число прибыльных сделок

    """

    return ((df['<DEAL_RESULT>']) * np.where(df['<DEAL_DIRECTION>'] == 'SHORT', 1, np.nan)).mean()


if __name__ == '__main__':
    # Здесь нужно заменить названия файлов и директорий на свои
    my_data_folder = "/home/basil/Documents/findata/customs/"
    my_data_filename = "asset_with_indicators2_with_positions_with_deals"

    try:
        df = pd.read_csv(my_data_folder + my_data_filename + ".csv")
    except UnicodeDecodeError:
        print("Cannot parse a file. Perhaps some wrong character in the file?")
    except TypeError:
        print("The computation began. but there's some error in a file. Check the data inside")
    except OSError:
        print("No such file or cannot read/write. Make sure everything is ok about this.")
    except KeyError:
        print("Error while parsing a file by pandas. "
              "Make sure the file is a consistent .csv and the delimiter is correct")
    else:
        print('1. total_strategy_income ', calculate_total_profit(df))
        print('2. max_portfolio_value ', calculate_max_profit(df))
        print('3. deepest_drawdown ', calculate_deepest_drawdown(df))
        print('4. deals_count ', count_deals(df))
        print('5. long_deals_count ', count_long_deals(df))
        print('6. short_deals_count ', count_short_deals(df))
        print('7. good_deals ', count_good_deals(df))
        print('8. bad_deals ', count_bad_deals(df))
        print('9. average_profit ', calculate_average_profit(df))
        print('10. good_long_deals', count_good_long_deals(df))
        print('11. bad_long_deals', count_bad_long_deals(df))
        print('12. average_long_deal', calculate_average_long_deal_profit(df))
        print('13. good_short_deals', count_good_short_deals(df))
        print('14. bad_short_deals', count_bad_short_deals(df))
        print('15. average_short_deal', calculate_average_short_deal_profit(df))

        logfile = open(my_data_folder + my_data_filename + '_stats.txt', 'w')
        logfile.write('1. total_strategy_income ' + str(calculate_total_profit(df)) + '\n')
        logfile.write('2. max_portfolio_value ' + str(calculate_max_profit(df)) + '\n')
        logfile.write('3. deepest_drawdown ' + str(calculate_deepest_drawdown(df)) + '\n')
        logfile.write('4. deals_count ' + str(count_deals(df)) + '\n')
        logfile.write('5. long_deals_count ' + str(count_long_deals(df)) + '\n')
        logfile.write('6. short_deals_count '+ str(count_short_deals(df)) + '\n')
        logfile.write('7. good_deals ' + str(count_good_deals(df)) + '\n')
        logfile.write('8. bad_deals ' + str(count_bad_deals(df)) + '\n')
        logfile.write('9. average_profit ' + str(calculate_average_profit(df)) + '\n')   
        logfile.write('10. good_long_deals'+ str(count_good_long_deals(df)) + '\n')
        logfile.write('11. bad_long_deals' + str(count_bad_long_deals(df)) + '\n')
        logfile.write('12. average_long_deal' + str(calculate_average_long_deal_profit(df)) + '\n')
        logfile.write('13. good_short_deals' + str(count_good_short_deals(df)) + '\n')
        logfile.write('14. bad_short_deals' + str(count_bad_short_deals(df)) + '\n')
        logfile.write('15. average_short_deal' + str(calculate_average_short_deal_profit(df)) + '\n')
        
        logfile.close()
