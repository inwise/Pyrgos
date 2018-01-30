from numpy import mean, sqrt
import option

source = open('C:\\findata\\deal_rts_options_extended_ba_vix.csv', 'r')
dict_sigma = {}

for line in source:
    splitted = line.split(',')
    if splitted[0] != 'day':
        key = splitted[0] + '/' + splitted[4] + '/' + splitted[5]
        sigma_arr = []
        if dict_sigma.keys().__contains__(key):
            sigma_arr = dict_sigma.get(key)
        sigma_arr.append(float(splitted[9]))
        dict_sigma[key] = sigma_arr

source.close()

source = open('C:\\findata\\deal_rts_options_extended_ba_vix.csv', 'r')
dest = open('C:\\findata\\deal_rts_options_extended_avg_volatility.csv', 'w')
dest_headers = ['day', 'strike', 'price', 'expiration', 'expiration_month', 'option_type', 'ba', 'vix', 'avg_daily_price', 'volatility', 'avg_daily_volatility', 'loss_function']
dest.write(','.join(dest_headers) + '\n')

for line in source:
    splitted = line.split(',')
    if splitted[0] != 'day':
        key = splitted[0] + '/' + splitted[4] + '/' + splitted[5]
        sigma_arr = dict_sigma.get(key)
        sigma_avg = mean(sigma_arr)

        option_type = splitted[5].replace('\n', '')

        s = int(float(splitted[6].replace('\n', '')))
        t1 = float(splitted[3].replace('\n', ''))
        k = int(float(splitted[1].replace('\n', '')))
        avg_price = float(splitted[8].replace('\n', ''))

        if s > 0 and t1 > 0 and k > 0 and sigma_avg > 0:
             if option_type == 'P':
                 loss_function = str(sqrt((option.euro_option_put_pricing(s, 0, t1, sigma_avg, 0, k) - avg_price)**2))
             else:
                 loss_function = str(sqrt((option.euro_option_call_pricing(s, 0, t1, sigma_avg, 0, k) - avg_price)**2))

             line += ',' + str(sigma_avg) + ',' + loss_function
             dest.write(line.replace('\n', '') + '\n')

source.close()
dest.close()