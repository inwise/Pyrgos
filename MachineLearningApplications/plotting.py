import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
"""
Этот файл содержит необходимые для рисования
графиков методы.

1. Метод, который рисует один из столбцов напротив даты
2. Метод, который рисует производительность портфеля
3. Метод, который располагает один график под другим

"""


def draw_column(df, source_filename, colname, label='label', color='blue'):
    """
    Рисует график любой из колонок в датафрейме.

    :param  df: датафрейм с данными
            colname: имя колонки, которую мы хотим нарисовать
            source_filename: имя файла, который обрабатывается
            colname: имя колонки, которая окажется нарисованной
    :return: 0
    """

    x = df['<DATE>'].values
    y = df[colname].values
    # plot data
    plt.plot(range(len(x)), y, color=color, label=label)
    plt.xticks(range(len(x))[0::100], [], size='xx-small', rotation=45)
    # plt.xlabel('Дата')
    plt.ylabel(colname)
    plt.legend(loc='upper left', prop={'size': 9})
    plt.tight_layout()

    # plt.savefig(source_filename + colname + '_plot.jpg', dpi=300)
    # plt.show()
    return 0


def draw_strategy_performance(df, source_filename, label='stratname'):
    """
    Рисует график доходности по портфелю.

    :param df: Датафрейм с данными о <PERFORMANCE> и <CLOSE>
           dest_filename: файл, в который мы хотим сохранить картинку для дальнейшего использования
    :return: 0
    """
    x = df['<DATE>'].values
    perf_data = df['<PERFORMANCE>']
    perf_data[0] = 0
    perf_data.fillna(method='ffill', inplace='True')

    mask_good = np.where(perf_data.values > 0, 1, np.nan)
    mask_bad = np.where(perf_data.values <= 0, 1, np.nan)
    y_good = mask_good * perf_data.values
    y_bad = mask_bad * perf_data.values

    # plotting
    plt.fill_between(range(len(x)), y_good, color='g', label='strategy')
    plt.fill_between(range(len(x)), y_bad, color='r')
    plt.plot(range(len(x)), get_buy_and_hold_column(df), color='b', label='buy_and_hold')

    plt.xticks(range(len(x))[0::100], x[0::100], size='xx-small', rotation=45)

    plt.xlabel('date')
    plt.ylabel(label)
    plt.legend(loc='upper left', prop={'size': 9})
    plt.axis('on')
    plt.grid()

    plt.tight_layout()
    plt.savefig(source_filename + '_performance_plot.jpg', dpi=300)
    # plt.show()
    return 0


def draw_multiplot(df, source_filename, label='strat'):
    """
    Рисует все нужные графики с информацией об активе.

    :param df: Датафрейм с данными о <PERFORMANCE> и <CLOSE>
           dest_filename: файл, в который мы хотим сохранить картинку для дальнейшего использования
    :return: 0
    """
    plt.subplot(4, 1, 1)
    draw_column(df, source_filename, '<CLOSE>', df['<TICKER>'].values[0], color='red')
    plt.subplot(4, 1, 2)
    draw_column(df, source_filename, '<RSI>', 'rsi', color='green')
    plt.subplot(4, 1, 3)
    draw_column(df, source_filename, '<VOL>', 'volatility', color='blue')
    plt.subplot(4, 1, 4)
    draw_strategy_performance(df, source_filename, label)

    plt.savefig(source_filename + '_multiplot.jpg', dpi=900)
    plt.show()
    return 0


def get_buy_and_hold_column(df):
    """
    Метод возвращает доход по стратегии buy-and-hold,
    нужен как бенчмарк

    :return: массив выплат по b&h
    """
    close_array = df['<CLOSE>'].values
    return [elem - close_array[0] for elem in close_array]


if __name__ == '__main__':
    # Здесь нужно заменить названия файлов и директорий на свои
    my_data_folder = "/home/basil/Documents/findata/customs/"
    my_data_filename = "ALRS_RSI_with_strategy"
    # 1 strategy markup
    df = pd.read_csv(my_data_folder + my_data_filename + ".csv")
    column_name = '<VOL>'
    source_filename = my_data_folder + my_data_filename
    # draw_strategy_performance(df, my_data_folder + my_data_filename + column_name + '_plot.jpg')
    draw_multiplot(df, source_filename)
