import pandas as pd
import numpy as np
"""
Для того, чтобы посчитать прибыль по стратегии, нужна стратегия. Чтобы добавить её на "голый" файл, предстоит
пользоваться другими скриптами. В качестве вероятного кандидата - position_maker.py, который есть в этом же проекте
Здесь содержится несколько полезных
функций.

Нужен .csv файл на входе, который содержит две колонки - "<CLOSE>" и "<POSITION>"
В колонке CLOSE содержатся числа (цены закрытия). Для простоты предположим, что они там всегда.
В колонке POSITION возможны варианты: BUY, SELL, SHORT, COVER. Ну и nan или пустота, если ничего не происходит.

Скрипт содержит методы обращения со стратегией - проверка её валидности и сбор одной тестовой метрики - прибыли.
Проверка валидности нужна для
того, чтобы отсеять то, что мы не можем обработать - для простоты предполагаем, что мы можем держать только одну позицию
по одному активу. То есть либо BUY-SELL, либо SHORT-COVER, за открытием следует закрытие.

После запуска этого скрипта файл должен дополняться:
1. колонкой "DEAL_DIRECTION", где указывается направление сделки
2. колонкой "DEAL_RESULT", где указывается прибыль по сделке
3. колонкой "PERFORMANCE", где описывается состояние портфеля в виде накопленной суммы
"""


def check_strategy_validity(df):
    """
    Первый и самый громоздкий метод, который выясняет, содержится ли валидная стратегия в столбце POSITION
    :param df: датафрейм, который содержит нужные колонки "CLOSE" и "POSITION"
    :return: строка, сообщение о валидности ("ok") или некорректности стратегии (их несколько)
    """
    positions = df['<POSITION>'].values

    last_position_type = 'NONE'
    validity_message = 'ok'
    is_last_position_closed = True

    index = 0
    while validity_message == 'ok' and index <= len(positions) - 1:
        if is_last_position_closed:
            if positions[index] == 'BUY':
                last_position_type = 'LONG_POSITION'
                is_last_position_closed = False
            elif positions[index] == 'SELL':
                validity_message = 'sell with no buy on ' + str(index+2)
            elif positions[index] == 'SHORT':
                last_position_type = 'SHORT_POSITION'
                is_last_position_closed = False
            elif positions[index] == 'COVER':
                validity_message = 'cover with no short on ' + str(index+2)
            else:
                pass
        else:
            if last_position_type == 'LONG_POSITION':
                if positions[index] == 'BUY':
                    validity_message = 'buy after long, double opening on ' + str(index+2)
                elif positions[index] == 'SELL':
                    is_last_position_closed = True
                elif positions[index] == 'SHORT':
                    validity_message = 'short on opened long, conflicting opening on ' + str(index+2)
                elif positions[index] == 'COVER':
                    validity_message = 'cover attempt on opened long, conflicting closing on ' + str(index+2)
                else:
                    pass

            elif last_position_type == 'SHORT_POSITION':
                if positions[index] == 'BUY':
                    validity_message = 'long after short, conflicting opening on ' + str(index+2)
                elif positions[index] == 'SELL':
                    validity_message = 'sell attempt on opened short, conflicting opening on ' + str(index+2)
                elif positions[index] == 'SHORT':
                    validity_message = 'short selling during active short, conflicting opening on ' + str(index+2)
                elif positions[index] == 'COVER':
                    is_last_position_closed = True
                else:
                    pass
        index += 1
    return validity_message


def calculate_strategy_profit(df):
    """
    Этот метод выясняет, сколько мы заработали на стратегии. Просто складываем прибыли в зависимости от того, короткая
    или длинная позиция была занята по сделке.
    Здесь важно, чтобы в файле был <POSITION> и <CLOSE>, иначе ничего не посчитается.
    Нужно ещё, чтобы была правильная стратегия, но метод для проверки стратегии есть ниже.

    :param df: датафрейм, который содержит нужные колонки "CLOSE" и "POSITION"
    :return: прибыль по стратегии, число
    """
    positions = df['<POSITION>'].values
    prices = df['<CLOSE>'].values
    strategy_profit = 0.0
    index = 0
    is_last_position_closed = True

    while index <= len(positions) - 1:
        if is_last_position_closed:
            if positions[index] == 'BUY':
                last_position_type = 'LONG_POSITION'
                is_last_position_closed = False
                last_buy_index = index
            elif positions[index] == 'SHORT':
                last_position_type = 'SHORT_POSITION'
                is_last_position_closed = False
                last_short_index = index
            else:
                pass
        else:
            if last_position_type == 'LONG_POSITION':
                if positions[index] == 'SELL':
                    strategy_profit += prices[index] - prices[last_buy_index]
                    is_last_position_closed = True
                else:
                    pass
            elif last_position_type == 'SHORT_POSITION':
                if positions[index] == 'COVER':
                    is_last_position_closed = True
                    strategy_profit += prices[last_short_index] - prices[index]
                else:
                    pass
        index += 1
    return strategy_profit


def collect_deals(df):
    """
    Метод, который собирает информацию о каждой проведённой сделке, чтобы потом можно было добавить её в конец файла
    :param df: датафрейм, который содержит нужные колонки "CLOSE" и "POSITION"
    :return: список(массив) вида [прибыль сделки, индекс (номер т.е позиция) сделки, тип занятой позиции]
    """
    positions = df['<POSITION>'].values
    prices = df['<CLOSE>'].values
    deals = []  # deal_result, deal_place, position type
    index = 0
    is_last_position_closed = True

    while index <= len(positions) - 1:
        if is_last_position_closed:
            if positions[index] == 'BUY':
                last_position_type = 'LONG_POSITION'
                is_last_position_closed = False
                last_buy_index = index
            elif positions[index] == 'SHORT':
                last_position_type = 'SHORT_POSITION'
                is_last_position_closed = False
                last_short_index = index
            else:
                pass
        else:
            if last_position_type == 'LONG_POSITION':
                if positions[index] == 'SELL':
                    deal_result = prices[index] - prices[last_buy_index]
                    is_last_position_closed = True
                    deals.append([deal_result, index, 'LONG'])
                else:
                    pass
            elif last_position_type == 'SHORT_POSITION':
                if positions[index] == 'COVER':
                    deal_result = prices[last_short_index] - prices[index]
                    is_last_position_closed = True
                    deals.append([deal_result, index, 'SHORT'])
                else:
                    pass
        index += 1
    return np.array(deals)


if __name__ == '__main__':

    # Здесь нужно заменить названия файлов и директорий на свои
    my_data_folder = "/home/basil/Documents/findata/customs/options_paper/"
    from os import listdir
    print(listdir(my_data_folder))
    my_data_filename = "BA"
    #my_data_filename = "BA.csv"
    df = pd.read_csv(my_data_folder + my_data_filename + ".csv")
    print(df.head())

    try:
        df = pd.read_csv(my_data_folder + my_data_filename + ".csv")
        validity_info = check_strategy_validity(df)
        if validity_info == 'ok':
            profit = calculate_strategy_profit(df)
            print("Strategy_total_profit: %.2f rub" % profit)
            deals = collect_deals(df)
            portfolio_performance = []
            deal_results = [float(info[0]) for info in deals]
            deal_positions = [int(info[1]) for info in deals]
            deal_directions = [info[2] for info in deals]
            buy_and_hold_profit = df['<CLOSE>'].values[-1] - df['<CLOSE>'].values[0]
            print("Buy_and_hold_profit: %.2f rub" % buy_and_hold_profit)
            deals_info_frame = pd.DataFrame({'<DEAL_RESULT>': deal_results, '<DEAL_DIRECTION>': deal_directions},
                                            index=[deal_positions])
            df = df.join(deals_info_frame)
            df['<PERFORMANCE>'] = df['<DEAL_RESULT>'].cumsum()
        else:
            print('The strategy in incorrect, reason: ' + validity_info)
    except UnicodeDecodeError:
        print("Cannot parse a file. Perhaps some wrong character in the file?")
    except OSError:
        print("No such file or cannot read/write. Make sure everything is ok about this.")
    except KeyError:
        print("Error while parsing a file by pandas. "
              "Make sure the file is a consistent .csv and the delimiter is correct")
    else:
        print(df.head())
        df.to_csv(my_data_folder + my_data_filename + "_with_deals.csv")
