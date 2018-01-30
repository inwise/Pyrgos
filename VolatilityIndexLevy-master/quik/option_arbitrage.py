# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

import datetime

class option_data:
    #класс данных опциона

    def __init__(self,mat_date,strike,put_last_price,put_buy,put_sell,call_last_price,call_buy,call_sell):
        self.strike=strike
        self.put_last_price=put_last_price
        self.put_buy=put_buy
        self.put_sell=put_sell
        self.call_last_price=call_last_price
        self.call_buy=call_buy
        self.call_sell=call_sell
        self.mat_date=mat_date

    def check_validity(self):
        #проверяет есть ли все котировки bid, ask
        if self.put_buy==0 or self.put_sell==0 or self.call_buy==0 or self.call_sell==0:
            return False
        return True

def rub_to_points(rub):
    #переводит рубли в пункты
    return int(rub/67.0*50.0)

def test_arbitrage_oppportunities(option_dict,rtsi,rtsi_sell,rtsi_buy):
    #проверяет выполняется ли условия отсутствия арбитража у опционов
    global sended
    delta=int(8.0/37.0*50.0)
    mat_dates=option_dict.keys()
    if len(mat_dates)<1:
        return
    mat_dates.sort()
    mat_date=mat_dates[0]
    opt_dict=option_dict[mat_date]
    strikes=opt_dict.keys()
    strikes.sort()
    if len(strikes)<3:
        return
    now_str=datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    for i in range(len(strikes)):
        opt_data_i=opt_dict[strikes[i]]

        #проверка условий call-put паритета
        if opt_data_i.call_buy!=0 and opt_data_i.put_sell!=0:
            profit=opt_data_i.call_buy-opt_data_i.put_sell-(rtsi_sell-strikes[i])
            if profit>rub_to_points(20):
                print "%s arbitrage call-put parity C-P>S-K strike = %g profit = %g"%(now_str,strikes[i],profit)
        if opt_data_i.call_sell!=0 and opt_data_i.put_buy!=0:
            profit=rtsi_buy-strikes[i]-(opt_data_i.call_sell-opt_data_i.put_buy)
            if profit>rub_to_points(20):
                print "%s arbitrage call-put parity C-P<S-K strike = %g profit = %g"%(now_str,strikes[i],profit)

        #проверка условий раннего исполнения
        if strikes[i]<rtsi:
            if opt_data_i.call_sell!=0:
                if opt_data_i.call_sell<rtsi-strikes[i]:
                    print "%s arbitrage early exercise call strike = %g"%(now_str,strikes[i],)
        else:
            if opt_data_i.put_sell!=0:
                if opt_data_i.put_sell<strikes[i]-rtsi:
                    print "%s arbitrage early exercise put strike = %g"%(now_str,strikes[i],)

        #проверка условий монотонности
        if i>0:
            opt_data_i1=opt_dict[strikes[i-1]]
            if opt_data_i1.call_sell!=0 and opt_data_i.call_buy!=0:
                if opt_data_i1.call_sell+2*delta<=opt_data_i.call_buy:
                    print "%s arbitrage monotonic opportunity call strikes = %g, %g"%(now_str,strikes[i-1],strikes[i],)
            if opt_data_i.put_sell!=0 and opt_data_i1.put_buy!=0:
                if opt_data_i1.put_buy>=opt_data_i.put_sell+2*delta:
                    print "%s arbitrage monotonic opportunity put strikes = %g, %g"%(now_str,strikes[i-1],strikes[i],)

        #проверка условий выпуклости
        if i>1:
            opt_data_i2=opt_dict[strikes[i-2]]
            if opt_data_i1.call_buy!=0 and opt_data_i2.call_sell!=0 and opt_data_i.call_sell!=0:
                profit=2*opt_data_i1.call_buy-(opt_data_i2.call_sell+opt_data_i.call_sell)
                if profit>=rub_to_points(24):
                    print "%s arbitrage convex opportunity call strike = %g profit = %g"%(now_str,strikes[i-1],profit,)
                    '''
                    if (not sended) and valid_time():
                        pcode_i=get_option_code(opt_data_i)
                        pcode_i1=get_option_code(opt_data_i1)
                        pcode_i2=get_option_code(opt_data_i2)
                        o=order(True,0,opt_data_i.call_sell,1)
                        o1=order(False,0,opt_data_i1.call_buy,2)
                        o2=order(True,0,opt_data_i2.call_sell,1)
                        eng.set_order(pcode_i,o)
                        eng.set_order(pcode_i2,o2)
                        eng.set_order(pcode_i1,o1)
                        sended=True
                    '''
            if opt_data_i1.put_buy!=0 and opt_data_i2.put_sell!=0 and opt_data_i.put_sell!=0:
                profit=2*opt_data_i1.put_buy-(opt_data_i2.put_sell+opt_data_i.put_sell)
                if profit>=rub_to_points(24):
                    print "%s arbitrage convex opportunity put strike = %g profit = %g"%(now_str,strikes[i-1],profit,)
                    '''
                    if (not sended) and valid_time():
                        pcode_i=get_option_code(opt_data_i,False)
                        pcode_i1=get_option_code(opt_data_i1,False)
                        pcode_i2=get_option_code(opt_data_i2,False)
                        o=order(True,0,opt_data_i.put_sell,1)
                        o1=order(False,0,opt_data_i1.put_buy,2)
                        o2=order(True,0,opt_data_i2.put_sell,1)
                        eng.set_order(pcode_i,o)
                        eng.set_order(pcode_i2,o2)
                        eng.set_order(pcode_i1,o1)
                        sended=True
                    '''
