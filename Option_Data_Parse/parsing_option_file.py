# by Grechko

import shelve
from datetime import datetime


def to_datetime(date):
    # преобразует ключевую строку в дату-время
    return datetime.strptime(date, "%d.%m.%Y %H:%M:%S")

d = shelve.open('c:\\findata\\options', 'r')

dates = d.keys()
dates.sort(key=lambda x: to_datetime(x))
# дата и время это ключ словаря d
print(dates[0])
print(dates[-1])

# первая запись
print(d[dates[0]])