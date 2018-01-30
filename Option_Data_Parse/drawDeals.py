import matplotlib.pyplot as plt
from os import listdir

sourceDirPaths = ['C:\\findata\\rts_call_options_by_days\\', 'C:\\findata\\rts_put_options_by_days\\']

for i in range(2):
    for fileName in listdir(sourceDirPaths[i]):
        if fileName.__contains__('.csv'):
            source = open(sourceDirPaths[i] + fileName, 'r')

            strikeVectorBlue = []
            priceVectorBlue = []
            strikeVectorRed = []
            priceVectorRed = []
            strikeVectorGreen = []
            priceVectorGreen = []
            strikeVectorYellow = []
            priceVectorYellow = []
            strikeVectorCyan = []
            priceVectorCyan = []

            for line in source:
                splitted = line.split(',')
                if splitted[0] != 'strike':
                    if splitted[3] == '06\n':
                        strikeVectorBlue.append(splitted[0])
                        priceVectorBlue.append(splitted[1])
                    elif splitted[3] == '07\n':
                        strikeVectorRed.append(splitted[0])
                        priceVectorRed.append(splitted[1])
                    elif splitted[3] == '08\n':
                        strikeVectorGreen.append(splitted[0])
                        priceVectorGreen.append(splitted[1])
                    elif splitted[3] == '09\n':
                        strikeVectorYellow.append(splitted[0])
                        priceVectorYellow.append(splitted[1])
                    elif splitted[3] == '12\n':
                        strikeVectorCyan.append(splitted[0])
                        priceVectorCyan.append(splitted[1])
                    else:
                        print(splitted[3])

            plt.plot(strikeVectorBlue, priceVectorBlue, 'bo')
            plt.plot(strikeVectorRed, priceVectorRed, 'ro')
            plt.plot(strikeVectorGreen, priceVectorGreen, 'go')
            plt.plot(strikeVectorYellow, priceVectorYellow, 'yo')
            plt.plot(strikeVectorCyan, priceVectorCyan, 'co')

            plt.xlabel('Strike')
            plt.ylabel('Price')

            bo = plt.Line2D(range(10), range(10), marker='o', color='b')
            ro = plt.Line2D(range(10), range(10), marker='o', color='r')
            go = plt.Line2D(range(10), range(10), marker='o', color='g')
            yo = plt.Line2D(range(10), range(10), marker='o', color='y')
            co = plt.Line2D(range(10), range(10), marker='o', color='c')

            plt.legend((bo, ro, go, yo, co), ('06', '07', '08', '09', '12'), numpoints=1, loc=1,  borderaxespad=0., fontsize=8)

            day = fileName.replace('price_day_', '').replace('.csv', '')
            plt.title('RTS option deals ' + day[6:8] + '.' + day[4:6] + '.' + day[0:4])

            plt.savefig(sourceDirPaths[i] + fileName.replace('.csv', '.png'))
            plt.close()
            source.close()

