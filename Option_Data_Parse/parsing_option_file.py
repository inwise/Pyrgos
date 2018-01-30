# by Grechko

import shelve
from datetime import datetime


def to_datetime(date):
    # ����������� �������� ������ � ����-�����
    return datetime.strptime(date, "%d.%m.%Y %H:%M:%S")

d = shelve.open('c:\\findata\\options', 'r')

dates = d.keys()
dates.sort(key=lambda x: to_datetime(x))
# ���� � ����� ��� ���� ������� d
print(dates[0])
print(dates[-1])

# ������ ������
print(d[dates[0]])