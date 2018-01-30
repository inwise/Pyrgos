import numpy as np
import pandas as pd
"""
Этот скрипт читает файл с данными об индикаторах и ценах закрытия и формирует стратегию, основанную на предсказаниях
леса-регрессора.

Формат нужного файла - csv
Необходимые колонки внутри:
<CLOSE>
<FOREST_CLOSE>
Он формирует новый файл, который содержит информацию о том, когда входить в сделки и выходить
из них, согласно торговой стратегии.
Правила стратегии следующие

BUY
<FOREST_CLOSE> > <CLOSE>

SELL
Открыта позиция BUY &
<FOREST_CLOSE> < <CLOSE>

SHORT
<FOREST_CLOSE> < <CLOSE>

COVER
Открыта позиция SHORT &
(
<FOREST_CLOSE> > <CLOSE>)
"""

def make_positions(df):
    """
    Метод, который открывает и закрывает сделки, в зависимости от состояния индикаторов.
    Нужные индикаторы:
    <RSI>, <VOL>, <DI+>, <DI->

    Правила стратегии следующие
    BUY
    <FOREST_CLOSE> > <CLOSE>

    SELL
    Открыта позиция BUY &
    <FOREST_CLOSE> < <CLOSE>

    SHORT
    <FOREST_CLOSE> < <CLOSE>

    COVER
    Открыта позиция SHORT &
    (
    <FOREST_CLOSE> > <CLOSE>)

    :param df: датафрейм, который содержит нужные колонки <RSI>, <VOL>, <DI+>, <DI->
    :return: список(массив) из позиций, N/A там, где позиций нет
    """
    prices = df['<CLOSE>'].values
    forest_prices = df['<FOREST_CLOSE>'].values
    positions = []
    index = 1  # умышленно пропускаем начальный индекс, в первый день можно только анализировать, но нельзя открываться
    is_last_position_closed = True
    last_position_type = 'NO_POSITION_YET'

    while index <= len(prices) - 1:
        if is_last_position_closed:
            # BUY opening: <FOREST_CLOSE> > <CLOSE> и нет открытых позиций
            if forest_prices[index-1] > prices[index-1]:
                positions.append('BUY')
                last_position_type = 'LONG_POSITION'
                is_last_position_closed = False
            # SHORT opening: <FOREST_CLOSE> < <CLOSE> и нет открытых позиций
            if forest_prices[index-1] < prices[index-1]:
                positions.append('SHORT')
                last_position_type = 'SHORT_POSITION'
                is_last_position_closed = False
            else:
                positions.append(np.nan)
        elif not is_last_position_closed:  # если позиция не закрыта (открыта)
            if last_position_type == 'LONG_POSITION':
                # SELL CLOSE: <FOREST_CLOSE> < <CLOSE>
                if forest_prices[index-1] < prices[index-1]:
                    positions.append('SELL')
                    is_last_position_closed = True
                else:
                    positions.append(np.nan)
            elif last_position_type == 'SHORT_POSITION':
                # COVER CLOSE <FOREST_CLOSE> > <CLOSE>
                if forest_prices[index-1] > prices[index-1]:
                    positions.append('COVER')
                    is_last_position_closed = True
                else:
                    positions.append(np.nan)
        index += 1
    return np.array(positions)


if __name__ == '__main__':
    # Здесь нужно заменить названия файлов и директорий на свои
    my_data_folder = "/home/basil/Documents/findata/customs/"
    my_data_filename = "asset_with_indicators2"
    try:
        df = pd.read_csv(my_data_folder + my_data_filename + ".csv", delimiter=',')
        positions_col = make_positions(df)
        positions_frame = pd.DataFrame({'<POSITION>': positions_col})
        df = df.join(positions_frame)
    except UnicodeDecodeError:
        print("Cannot parse a file. Perhaps some wrong character in the file?")
    except OSError:
        print("No such file or cannot read/write. Make sure everything is ok about this.")
    except KeyError:
        print("Error while parsing a file by pandas. "
              "Make sure the file is a consistent .csv and the delimiter is correct")
    else:
        print(df.head())
        try:
            df.to_csv(my_data_folder + my_data_filename + "_with_positions.csv")
        except OSError:
            print("No such file or cannot read/write. Make sure everything is ok about this.")

