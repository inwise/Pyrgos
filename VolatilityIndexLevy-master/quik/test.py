# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

from quik import start_quik_dde_server
from option_arbitrage import option_data,test_arbitrage_oppportunities

def quik_data_to_option_dict(data):
    #конвертирует данные из quik в option_dict
    calls=dict()
    puts=dict()
    for d in data[2:]:
        mat_date=d[4]
        opts=calls
        if d[6]=='Put':
            opts=puts
        if not opts.has_key(mat_date):
            opts[mat_date]=dict()
        opts[mat_date][d[5]]=d
    if len(calls)!=len(puts):
        print 'Count of calls and puts is not same'
        return None
    option_dict=dict()
    for mat_date in calls.keys():
        for strike in calls[mat_date].keys():
            call=calls[mat_date][strike]
            put=puts[mat_date][strike]
            o=option_data(mat_date,strike,put[3],put[1],put[2],call[3],call[1],call[2])
            if not option_dict.has_key(mat_date):
                option_dict[mat_date]=dict()
            option_dict[mat_date][strike]=o
    return option_dict

def monitor(data):
    option_dict=quik_data_to_option_dict(data)
    rtsi=data[1]
    test_arbitrage_oppportunities(option_dict,rtsi[3],rtsi[2],rtsi[1])


start_quik_dde_server('DDE',monitor)