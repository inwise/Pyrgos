from numpy import mean, sqrt
import option


class Option:
    def __init__(self):
        self.strike = ''
        self.ba_price = ''
        self.vix = ''
        self.expiration = ''
        self.avg_price = ''
        self.volatility = ''
        self.avg_volatility = ''
        self.loss_function = ''

source = open('C:\\findata\\deal_rts_options_extended_ba_vix.csv', 'r')
dict_strike = {}
strike_all_arr = []

for line in source:
    splitted = line.split(',')
    if splitted[0] != 'day':
        key = splitted[0] + '/' + splitted[4] + '/' + splitted[5]
        strike_arr = []
        strike = splitted[1]

        if dict_strike.keys().__contains__(key):
            strike_arr = dict_strike.get(key)

        strike_arr.append(strike)
        dict_strike[key] = strike_arr

        if not strike_all_arr.__contains__(strike):
            strike_all_arr.append(strike)

source.close()

strike_result = sorted(strike_all_arr)

source = open('C:\\findata\\deal_rts_options_extended_avg_volatility.csv', 'r')
dest = open('C:\\findata\\dreamFile.csv', 'w')

dest_headers = ['Day', 'BA_price', 'VIX', 'Put/Call', 'Expiration', 'Expiration_Month']
for strike in strike_result:
    dest_headers.append('Strike_' + str(strike) + '_AVG_Daily_Price')

for strike in strike_result:
    dest_headers.append('Strike_' + str(strike) + '_Volatility')

dest_headers.append('AVG_Volatility')
dest_headers.append('Loss_Function')

dest.write(','.join(dest_headers) + '\n')
dict_opt = {}

for line in source:
    splitted = line.split(',')
    if splitted[0] != 'day':
        key = splitted[0] + '/' + splitted[4] + '/' + splitted[5]
        opt = Option()
        opt.strike = splitted[1]
        opt.ba_price = splitted[6]
        opt.vix = splitted[7]
        opt.expiration = splitted[3]
        opt.avg_price = splitted[8]
        opt.volatility = splitted[9]
        opt.avg_volatility = splitted[10]
        opt.loss_function = splitted[11]
        opt_arr = []
        if dict_opt.keys().__contains__(key):
            opt_arr = dict_opt.get(key)
            strike_added = 0
            for option in opt_arr:
                if option.strike == opt.strike:
                    strike_added = 1
            if strike_added == 0:
                opt_arr.append(opt)
        else:
            opt_arr.append(opt)
        dict_opt[key] = opt_arr

source.close()

for key in sorted(dict_opt.keys()):
    line = []
    splitted = key.split('/')
    data = splitted[0]
    exp_month = splitted[1]
    opt_type = splitted[2]
    opt_arr = dict_opt.get(key)
    ba_price = opt_arr[0].ba_price
    vix = opt_arr[0].vix
    expiration = opt_arr[0].expiration
    avg_volatility = opt_arr[0].avg_volatility
    loss_function = opt_arr[0].loss_function

    line.append(data)
    line.append(ba_price)
    line.append(vix)
    line.append(opt_type)
    line.append(expiration)
    line.append(exp_month)

    for strike in strike_result:
        strike_added = 0
        for opt in opt_arr:
            if str(int(strike)) == str(int(opt.strike)):
                line.append(opt.avg_price)
                strike_added = 1
        if strike_added == 0:
            line.append('')

    for strike in strike_result:
        strike_added = 0
        for opt in opt_arr:
            if str(int(strike)) == str(int(opt.strike)):
                line.append(opt.volatility)
                strike_added = 1
        if strike_added == 0:
            line.append('')

    line.append(avg_volatility)
    line.append(loss_function)
    dest.write(','.join(line).replace('\n', '') + '\n')

source.close()
dest.close()