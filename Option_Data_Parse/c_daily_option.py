__author__ = 'Rodochenko'

import datetime

class rtsDailyOption:
    def __init__(self, strike, opt_type, code, date, value):
        self.strike = strike
        self.code = code
        self.mat_date = self.get_mat_date_from_code(self)
        self.opt_type = opt_type
        self.date = date
        self.value = value

    def get_mat_date_from_code(self):
        code = self.code
        call_month_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        put_month_codes = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
        mat_date_month = "00"
        if code[-2] in call_month_codes:
            # retrieving month code
            mat_date_month = str(call_month_codes.index(code[-2]) + 1)
        elif code[-2] in put_month_codes:
            mat_date_month = str(put_month_codes.index(code[-2]) + 1)

        mat_date_year = "0000"
        if code[-1] == 1:
            mat_date_year = "2011"
        elif code[-1] == 2:
            mat_date_year = "2012"

        mat_date_day = "15"
        mat_day_moment_string = mat_date_year + mat_date_month + mat_date_day
        return mat_day_moment_string


def get_maturity_from_code(code):
    """getting expiration date from FORTS code"""

    call_month_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    put_month_codes = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    mat_date_month = "00"
    if code[-2] in call_month_codes:
        # retrieving month code
        mat_date_month = str(call_month_codes.index(code[-2]) + 1)
        if len(mat_date_month)<2:
            mat_date_month = '0' + mat_date_month
    elif code[-2] in put_month_codes:
        mat_date_month = str(put_month_codes.index(code[-2]) + 1).format()
        if len(mat_date_month)<2:
            mat_date_month = '0' + mat_date_month
    mat_date_year = "0000"
    if code[-1] == '1':
        mat_date_year = "2011"
    elif code[-1] == '2':
        mat_date_year = "2012"
    elif code[-1] == '3':
        mat_date_year = "2012"
    elif code[-1] == '4':
        mat_date_year = "2014"
    elif code[-1] == '5':
        mat_date_year = "2015"

    mat_date_day = "15"
    mat_day_moment_string = mat_date_year + mat_date_month + mat_date_day
    return mat_day_moment_string


def get_time_to_expiration(current_date_day, maturity_day):
    days_in_the_year = 248
    try:
        date0 = datetime.datetime.strptime(current_date_day, "%Y%m%d")
        date1 = datetime.datetime.strptime(maturity_day, "%Y%m%d")
        date_dif = date1-date0
        value = str(float(date_dif.days)/days_in_the_year)
    except:
        value = 0
    return value
