source = open('C:\\findata\\r_files\\rts_close_daily.csv', 'r')

dictCloseDaily = {}

for line in source:
    splitted = line.split(',')
    if not dictCloseDaily.keys().__contains__(splitted[1]):
        dictCloseDaily[splitted[1]] = splitted[2]

source.close()

source = open('C:\\findata\\deal_rts_options.csv', 'r')
dest = open('C:\\findata\\deal_rts_options_extended_with_close_daily.csv', 'w')

for line in source:
    splitted = line.split(',')
    if splitted[0] == 'SYMBOL':
        line = line.replace('\n', '') + ',BA_CLOSE\n'
    else:
        line = line.replace('\n', '') + ',' + dictCloseDaily.get(splitted[2][0:8])

    dest.write(line)

dest.close()

