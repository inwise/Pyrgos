from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# TODO: define x, y values correctly

df = pd.read_csv('calibration_big_output.csv', delimiter=',')
# print(df)
print(df.columns)
colnames_to_train_from = ['Ri',
                          'exp_time',
                          'RTSVX',
                          'NRTSVX',
                          # 'Put/Call',
                          'Strike_140000_AVG_Daily_Price',
                          'Strike_145000_AVG_Daily_Price',
                          'Strike_150000_AVG_Daily_Price',
                          'Strike_155000_AVG_Daily_Price',
                          'Strike_160000_AVG_Daily_Price',
                          'AVG_volatility',
                          # new approach
                          'h-kappa',
                          'h-rho',
                          'h-sigma',
                          'h-theta',
                          'h-v0'
                          # merton approach
                          ]

#heston

# colnames_to_predict = ['h-kappa', 'h-rho', 'h-sigma', 'h-theta', 'h-v0']

# merton

colnames_to_predict = ['m-sigma', 'm-delta', 'm-mu', 'm-lambda']

X = df[colnames_to_train_from].values[0:200]

y = df[colnames_to_predict].values[1:201]

# y = df['AVG_volatility'].values[1:201]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# случайный лес

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1, verbose=1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE forest train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 forest train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred, multioutput='uniform_average'),
        r2_score(y_test, y_test_pred, multioutput='uniform_average')))

# регрессия через svm
# SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
#     kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# svr_classifier = SVR(C=1.0, epsilon=0.2)
# svr_classifier.fit(X, y)
#
# y_train_pred = svr_classifier.predict(X_train)
# y_test_pred = svr_classifier.predict(X_test)
#
#
# print('MSE forest train: %.3f, test: %.3f' % (
#     mean_squared_error(y_train, y_train_pred),
#     mean_squared_error(y_test, y_test_pred)))
# print('R^2 forest train: %.3f, test: %.3f' % (
#     r2_score(y_train, y_train_pred, multioutput='uniform_average'),
#     r2_score(y_test, y_test_pred, multioutput='uniform_average')))
