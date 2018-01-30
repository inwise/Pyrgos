# ______________DRAWING________________
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/home/basil/Documents/findata/dreamFile.csv', delimiter=',')
strikes = [strike for strike in range(120000, 200000, 5000)]
strike_strings = ['Strike_' +str(strike) + '_Volatility' for strike in strikes]
# K = np.array([strikes])
imp_vols = df[strike_strings].values[11]

#
print(type(strikes[0]))
print(type(imp_vols[0]))
# plot result
plt.plot(strikes, imp_vols, 'g*')
plt.xlabel('Strike (K)')
plt.ylabel('Implied volatility')
plt.savefig('smile_RTS.png')
plt.show()
