import pandas as pd


def sample_data():
    """
    Just extracts a sample options data from a file
    :return: header, market datas in order: S0, K, Put/Call, option_price, r, T
    """
    # sample market data
    x = [x.split() for x in open('marketdata.txt')]
    header = x[0]
    market_datas = []
    for market_data in x[1:]:
        datarow = [float(market_data[0]), float(market_data[1]), market_data[2], float(market_data[3]),
                   float(market_data[4]), float(market_data[5])]
        market_datas.append(datarow)
    return header, market_datas


def sample_dreamfile():
    """
    Читает (пока только одну) строчку из файла DreamFile, который содержит данные по опционной доске.

    :return: header, market_data - возвращает заголовок и данные, в виде кортежа.
    Заголовок возвращается как pandas.indexes.base.Index
    Сами же данные - как list
    """
    # a small cut of dreamfile
    content = pd.read_csv('../dreamFile_cut.csv', delimiter=';')
    data = content.iloc[1]
    strikes = range(175000, 200000, 5000)  # их при желании можно заменить явным списком
    header = content.columns
    market_data = []
    for strike in strikes:
        s0 = data['BA_price']
        K = strike
        put_or_call = data['Put/Call']
        option_price = data['Strike_' + str(strike) +'_AVG_Daily_Price']
        r = 0.0  # наш актив - опционы на фьючерс, здесь можно не учитывать эти эффекты.
        T = data['Expiration']
        market_data.append([s0, K, put_or_call, option_price, r, T])
    return header, market_data

if __name__ == '__main__':
    print(sample_dreamfile())
    print(sample_data())
    print(type(sample_dreamfile()))
    print(type(sample_dreamfile()[0]))
    print(type(sample_dreamfile()[1]))
