# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

from quik import quik_dde_server
import redis
import cPickle as pickle

class quik_dde_server_oi(quik_dde_server):

    def __init__(self, topic_name):
        self.r=redis.Redis()
        #self.r.set_response_callback('GET',int)
        quik_dde_server.__init__(self,topic_name)

    def path(self,date):
        return self.topic_name+':'+date+':'

    def update(self, newdata, i0, j0):
        if len(newdata)==0:
            return
        date=newdata[0][2]
        last_i=self.r.get(self.path(date)+'last-i')
        if last_i is None:
            last_i=0
        else:
            last_i=int(last_i)
        if last_i>=i0:
            return
        pipe=self.r.pipeline()
        for d in newdata:
            pipe.rpush(self.path(date)+'values',pickle.dumps(d))
        pipe.set(self.path(date)+'last-i',i0+len(newdata)-1)
        pipe.execute()


qds=quik_dde_server_oi('OI')
qds.start()
