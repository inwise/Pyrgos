import numpy as np
import pandas as pd
"""
Этот скрипт читает файл с данными об индикаторах и ценах закрытия.
Формат нужного файла - csv
Необходимые колонки внутри:
<CLOSE>
<RSI>
<VOL>
<DI+>
<DI->
Он формирует новый файл, который содержит информацию о том, когда входить в сделки и выходить
из них, согласно торговой стратегии.
Правила стратегии следующие

BUY
RSI in [30,70] &
VOL < 45 &
DI+ > DI-

SELL
Открыта позиция BUY &
(
RSI > 70
VOL >= 45
DI+ < DI-
)

SHORT
RSI in [30,70] &
VOL < 45 &
DI+ < DI-

COVER
Открыта позиция SHORT &
(
RSI < 30
VOL >= 45
DI+ > DI-
)
"""

def make_positions(df):
    """
    Метод, который открывает и закрывает сделки, в зависимости от состояния индикаторов.
    Нужные индикаторы:
    <RSI>, <VOL>, <DI+>, <DI->

    Правила стратегии следующие
    BUY:
    ( RSI in [30,70] & VOL < 45 & DI+ > DI- ) и нет открытых позиций

    SELL:
    Открыта позиция BUY &
    (RSI > 70 or VOL >= 45 or DI+ < DI-)

    SHORT:
    ( RSI in [30,70] & VOL < 45 & DI+ < DI- ) и нет открытых позиций

    COVER:
    Открыта позиция SHORT &
    (RSI < 30 or VOL >= 45 or DI+ > DI-)

    :param df: датафрейм, который содержит нужные колонки <RSI>, <VOL>, <DI+>, <DI->
    :return: список(массив) из позиций, N/A там, где позиций нет
    """
    prices = df['<CLOSE>'].values
    RSIs = df['<RSI>'].values
    VOLs = df['<VOL>'].values
    DI_pluses = df['<DI+>'].values
    DI_minuses = df['<DI->'].values
    positions = []
    index = 1  # умышленно пропускаем начальный индекс, в первый день можно только анализировать, но нельзя открываться
    is_last_position_closed = True
    last_position_type = 'NO_POSITION_YET'

    while index <= len(prices) - 1:
        if is_last_position_closed:
            # BUY opening: ( RSI in [30,70] & VOL < 45 & DI+ > DI- ) и нет открытых позиций
            if 30 < RSIs[index-1] < 70 and VOLs[index-1] < 45\
                    and DI_pluses[index-1] > DI_minuses[index-1]:
                positions.append('BUY')
                last_position_type = 'LONG_POSITION'
                is_last_position_closed = False
            # SHORT opening: ( RSI in [30,70] & VOL < 45 & DI+ < DI- ) и нет открытых позиций
            if 30 < RSIs[index - 1] < 70 and VOLs[index - 1] < 45 \
                    and DI_pluses[index - 1] < DI_minuses[index - 1]:
                positions.append('SHORT')
                last_position_type = 'SHORT_POSITION'
                is_last_position_closed = False
            else:
                positions.append(np.nan)
        elif not is_last_position_closed:  # если позиция не закрыта (открыта)
            if last_position_type == 'LONG_POSITION':
                # SELL CLOSE: Открыта позиция BUY & (RSI > 70 or VOL >= 45 or DI+ < DI-)
                if RSIs[index - 1] >= 70 or VOLs[index-1] >= 45 or DI_pluses[index-1] < DI_minuses[index-1]:
                    positions.append('SELL')
                    is_last_position_closed = True
                else:
                    positions.append(np.nan)
            elif last_position_type == 'SHORT_POSITION':
                # COVER CLOSE Открыта позиция SHORT &(RSI < 30 or VOL >= 45 or DI + > DI -)
                if RSIs[index - 1] <= 30 or VOLs[index-1] >= 45 or DI_pluses[index-1] > DI_minuses[index-1]:
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

