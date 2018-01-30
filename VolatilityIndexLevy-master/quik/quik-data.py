# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

from quik import quik_dde_server
import redis
import cPickle as pickle

class quik_dde_server_trades(quik_dde_server):

    def __init__(self, topic_name):
        self.r=redis.Redis()
        #self.r.set_response_callback('GET',int)
        quik_dde_server.__init__(self,topic_name)

    def update(self, newdata, i0, j0):
        max_trade_id=self.r.get('max-trade-id')
        if max_trade_id is None:
            max_trade_id=0
            self.prev_id=0
        else:
            max_trade_id=int(max_trade_id)
            self.prev_id=max_trade_id
        nmax_trade_id=max_trade_id
        pipe=self.r.pipeline()
        for i in range(len(newdata)):
            try:
                id=int(newdata[i][1])
                if id<self.prev_id:
                    print "List of trades is not sorted by id, prev_id=",self.prev_id,' id=',id
                self.prev_id=id
            except:
                if i0>1:
                    print "Fail i0=",i0,' id=',newdata[i][1]
                continue
            if id>nmax_trade_id:
                if id-nmax_trade_id>1:
                    print "Missing trades between ",nmax_trade_id,' and ',id
                pcode=newdata[i][3]
                date=newdata[i][9]
                pipe.sadd('dates',date)
                pipe.sadd('pcodes:'+date,pcode)
                pipe.rpush('trades:'+date+':'+pcode,pickle.dumps(newdata[i]))
                #pipe.rpush('trades',newdata[i])
                nmax_trade_id=id
        if nmax_trade_id>max_trade_id:
            pipe.set('max-trade-id',nmax_trade_id)
        pipe.execute()


qds=quik_dde_server_trades('TRADES')
qds.start()
