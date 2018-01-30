from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
df['<UP_DOWN>'] = np.where(df['<LOG_INCREMENT>'] >= 0, 1, -1)

print(my_data_filename)

colnames_to_train_from = [
    # '<PER>', '<DATE>', '<TICKER>',
    '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>',
    '<VOLUME>', '<LOG_INCREMENT>', '<VOL>', '<RSI>', '<DI+>', '<DI->'
]

colnames_to_predict = ['<UP_DOWN>']
# индексация важна - при этом мы учимся на сегодняшних данных, а прогнозируем завтрашние
X = df[colnames_to_train_from].values[0:-1]
y = df[colnames_to_predict].values[1:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# случайный лес

forest = RandomForestClassifier(n_estimators=1000, verbose=1)
forest.fit(X_train, y_train.ravel())
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
y_total_pred = forest.predict(X)
print(df.last_valid_index())
print(len(y_total_pred))
# print(print(len(np.insert(y_total_pred, 0, df['<CLOSE>'][0]))))
df['<FOREST_UP_DOWN>'] = np.append(y_total_pred, df['<UP_DOWN>'][df.last_valid_index()])
# print(df.head())

print('MSE forest train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 forest train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred, multioutput='uniform_average'),
        r2_score(y_test, y_test_pred, multioutput='uniform_average')))

print('forest confusion train')
print(confusion_matrix(y_train, y_train_pred))
print('forest confusion test')
print(confusion_matrix(y_test, y_test_pred))

print('forest accuracy train')
print(accuracy_score(y_train, y_train_pred))
print('forest accuracy test')
print(accuracy_score(y_test, y_test_pred))

df.to_csv(my_data_folder + my_data_filename + ".csv")
