import matplotlib.pyplot as plt
from os import listdir

sourceDirPaths = ['C:\\findata\\rts_call_options_by_days_and_exp_month\\', 'C:\\findata\\rts_put_options_by_days_and_exp_month\\']

for i in range(2):
    for fileName in listdir(sourceDirPaths[i]):
        if fileName.__contains__('.csv'):
            source = open(sourceDirPaths[i] + fileName, 'r')
            expMonth = fileName.replace('price_day_and_exp_month_', '').replace('.csv', '').split('_')[1]

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
                    if expMonth == '06':
                        strikeVectorBlue.append(splitted[0])
                        priceVectorBlue.append(splitted[1])
                    elif expMonth == '07':
                        strikeVectorRed.append(splitted[0])
                        priceVectorRed.append(splitted[1])
                    elif expMonth == '08':
                        strikeVectorGreen.append(splitted[0])
                        priceVectorGreen.append(splitted[1])
                    elif expMonth == '09':
                        strikeVectorYellow.append(splitted[0])
                        priceVectorYellow.append(splitted[1])
                    elif expMonth == '12':
                        strikeVectorCyan.append(splitted[0])
                        priceVectorCyan.append(splitted[1])

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

            day = fileName.replace('price_day_and_exp_month_', '').replace('.csv', '')
            plt.title('RTS option deals ' + day[6:8] + '.' + day[4:6] + '.' + day[0:4])

            plt.savefig(sourceDirPaths[i] + fileName.replace('.csv', '.png'))
            plt.close()
            source.close()

