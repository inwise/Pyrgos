# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

import pandas as pd
from quik import quik_dde_server

class quik_dde_server_df(quik_dde_server):

    def __init__(self, topic_name, cols):
        self.cols=cols
        quik_dde_server.__init__(self,topic_name)

    def update(self, newdata, i0, j0):
        #if i0>1:
        #    ndf.index+=i0-1
        if self.data is None:
            self.data=pd.DataFrame(newdata,columns=self.cols[j0-1:])
            print self.data
        else:
            if i0-1>len(self.data):
                if len(newdata)>1:
                    ndf=pd.DataFrame(newdata,columns=self.cols[j0-1:])
                    ndf.index+=i0-1
                    self.data=self.data.append(ndf)
                    #self.data=self.data.append(ndf,ignore_index=True)
                elif len(newdata)==1:
                    print 'Index :',i0
                    print 'Len of data: ',len(self.data)
                    self.data.loc[i0-1]=newdata[0]
        print len(self.data)


qds=quik_dde_server_df('TRADES',('row','id','time','asset','price','count','volume','trade'))
qds.start()
