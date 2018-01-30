import matplotlib as plt
from c_daily_option import get_maturity_from_code, get_time_to_expiration
source = open('C:\\findata\\deal_rts_options.csv', 'r')
dest = open('C:\\findata\\deal_rts_options_extended.csv', 'w')
dest2 = open('C:\\findata\\deal_rts_options_extended_small.csv', 'w')
headers = source.readline().split(',')
headers_additional = ('NAME', 'STRIKE', 'MATURITY_NAME')
headers.extend(headers_additional)

# mending output a bit
for line in headers:
    if line.endswith('\n'):
        headers.remove(line)
        line = line.split('\n')[0]
        headers.append(line)
    elif line.startswith('#'):
        headers.remove(line)
        line = line.split('#')[1]
        headers.append(line)

d = {key: 'nan' for key in headers}

restricted_data = {}

dest_headers = ['day', 'strike', 'price', 'expiration', 'expiration_month', 'option_type']
dest.write(','.join(dest_headers) + '\n')

#dest.write(str(d.keys()) + '\n')

for line in source:
    splitted = line.split(',')
    d['SYMBOL'] = splitted[0]
    d['SYSTEM'] = splitted[1]
    d['MOMENT'] = splitted[2]
    d['ID_DEAL'] = splitted[3]
    d['PRICE_DEAL'] = splitted[4]
    d['VOLUME'] = splitted[6]
    d['OPEN_POS'] = splitted[5]
    d['DIRECTION'] = splitted[7][0]
    d['NAME'] = d['SYMBOL'][0:2]
    d['DAY'] = d['MOMENT'][0:8]
    d['STRIKE'] = d['SYMBOL'][2:len(d['SYMBOL'])-3]
    d['MATURITY_NAME'] = d['SYMBOL'][-2:]
    d['MATURITY_DATE'] = get_maturity_from_code(d['SYMBOL'])
    d['TIME_TO_EXPIRATION'] = get_time_to_expiration(d['DAY'], d['MATURITY_DATE'])

    restricted_data['day'] = d['DAY']
    restricted_data['option_type'] = d['SYSTEM']
    restricted_data['strike'] = d['STRIKE']
    restricted_data['price'] = d['PRICE_DEAL']
    restricted_data['expiration_month'] = d['MATURITY_DATE'][4:6]
    restricted_data['expiration'] = d['TIME_TO_EXPIRATION'].format('{8.f}')
    list_to_print = [restricted_data['day'], restricted_data['strike'], restricted_data['price'],
                     restricted_data['expiration'],  restricted_data['expiration_month'], restricted_data['option_type']]
    dest.write(','.join(list_to_print) + '\n')

source.close()
dest.close()


