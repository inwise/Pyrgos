import matplotlib as plt
from c_daily_option import get_maturity_from_code, get_time_to_expiration
from numpy import mean
import option

source_ba = open('C:\\findata\\BA.csv', 'r')
dict_ba = {}

for line in source_ba:
    splitted = line.split(',')
    dict_ba[splitted[2]] = splitted[4]

source_ba.close()

source_vix = open('C:\\findata\\RTSVX.txt', 'r')
dict_vix = {}

for line in source_vix:
    splitted = line.split(',')
    dict_vix[splitted[2]] = splitted[4]

source_vix.close()


source = open('C:\\findata\\deal_rts_options_extended.csv', 'r')
dict_strike = {}

for line in source:
    splitted = line.split(',')
    if splitted[0] != 'day':
        key = splitted[0] + '/' + splitted[1] + '/' + splitted[4] + '/' + splitted[5]
        strike_arr = []
        if dict_strike.keys().__contains__(key):
            strike_arr = dict_strike.get(key)
        strike_arr.append(float(splitted[2]))
        dict_strike[key] = strike_arr

source.close()

destPath = 'C:\\findata\\deal_rts_options_extended_ba_vix.csv'
dest_headers = ['day', 'strike', 'price', 'expiration', 'expiration_month', 'option_type', 'ba', 'vix', 'avg_daily_price', 'volatility']
dest = open(destPath, 'w')
dest.write(','.join(dest_headers) + '\n')

source = open('C:\\findata\\deal_rts_options_extended.csv', 'r')
for line in source:
    splitted = line.split(',')
    if splitted[0] != 'day':
         date = splitted[0]
         key = splitted[0] + '/' + splitted[1] + '/' + splitted[4] + '/' + splitted[5]
         avg_daily_price_arr = dict_strike.get(key)
         avg_daily_price = mean(avg_daily_price_arr)

         option_type = splitted[5].replace('\n', '')

         s = int(float(dict_ba.get(date).replace('\n', '')))
         tl = float(splitted[3].replace('\n',''))
         k = int(float(splitted[1].replace('\n', '')))
         o = float(splitted[2].replace('\n',''))

         if s > 0 and tl > 0 and k > 0 and o > 0:
             if option_type == 'P':
                 volatility = str(option.euro_option_put_volatility(s, 0, tl, 0, k, o))
             else:
                 volatility = str(option.euro_option_call_volatility(s, 0, tl, 0, k, o))

             line += ',' + dict_ba.get(date) + ',' + dict_vix.get(date) + ',' + str(avg_daily_price) + ',' + volatility
             dest.write(line.replace('\n', '') + '\n')

source.close()
dest.close()