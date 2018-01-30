from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
"""
В этом скрипте мы предскажем поведение цены <CLOSE>
при помощи регрессора "случайный лес"

Файл, который нужен на входе, должен содержать максимум доступной информации.

Тестирование проведём таким образом, чтобы разбить выборку на две, а лучше и более групп,
чтобы избежать оверфиттинга.

"""
my_data_folder = "/home/basil/Documents/findata/customs/"
my_data_filename = "SBER_RSI"
df = pd.read_csv(my_data_folder + my_data_filename + ".csv")
print(df.columns)

colnames_to_train_from = [
    # '<PER>', '<DATE>', '<TICKER>',
    '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>',
    '<VOLUME>', '<LOG_INCREMENT>', '<VOL>', '<RSI>', '<DI+>', '<DI->'
]

colnames_to_predict = ['<CLOSE>']
# индексация важна - при этом мы учимся на сегодняшних данных, а прогнозируем завтрашние
X = df[colnames_to_train_from].values[0:-1]
y = df[colnames_to_predict].values[1:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# случайный лес

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1, verbose=1)
forest.fit(X_train, y_train.ravel())
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
y_total_pred = forest.predict(X)
print(df.last_valid_index())
print(len(y_total_pred))
# print(print(len(np.insert(y_total_pred, 0, df['<CLOSE>'][0]))))
df['<FOREST_CLOSE>'] = np.append(y_total_pred, df['<CLOSE>'][df.last_valid_index()])
print(df['<FOREST_CLOSE>'])
# print(df.head())

print('MSE forest train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 forest train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred, multioutput='uniform_average'),
        r2_score(y_test, y_test_pred, multioutput='uniform_average')))

df.to_csv(my_data_folder + my_data_filename + ".csv")
