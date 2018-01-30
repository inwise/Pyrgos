# -*- coding: utf-8 -*-

__author__ = 'Александр Гречко'

import win32ui
import win32event
import pywin.mfc
import pywin.mfc.object
import dde
import struct
import re
import redis
import cPickle as pickle


def unpack_strings(b):
    # из буфера b pascal-like строк выделяет список строк
    i = 0
    fmt = ''
    while i < len(b):
        n = struct.unpack('B', b[i])[0]
        fmt += str(n + 1) + 'p'
        i += n + 1
    return [s.decode('cp1251') for s in struct.unpack(fmt, b)]


class QuikTopic(pywin.mfc.object.Object):
    # класс получает данные из Quik по DDE, парсит и отдает на обработку классу quik_dde_server

    def __init__(self, parent):
        self.parent = parent
        topic = dde.CreateTopic(self.parent.topic_name)
        topic.AddItem(dde.CreateStringItem(""))
        pywin.mfc.object.Object.__init__(self, topic)

    def Poke(self, a, b):
        # получение и обработка новых данных по DDE
        m = re.findall(r'\d+', a)
        if len(m) != 4:
            print 'Error ', m
            return
        x = [int(a) for a in m]
        # парсим a - диапазон ячеек, который передается
        # читаем первый блок tdtTable
        t, n = struct.unpack('HH', b[:4])
        if t != 16 or n != 4:
            print 'Error'
            return
        # кол-во строк и столбцов в таблице
        nrows, ncols = struct.unpack('HH', b[4:8])
        k = 8
        newdata = [[None] * ncols for i in range(nrows)]
        i = 0
        while i < nrows * ncols:
            t, n = struct.unpack('HH', b[k:k + 4])
            if t == 1:
                fmt = ''
                c = n / 8
                for j in range(c):
                    fmt += 'd'
                d = struct.unpack(fmt, b[k + 4:k + 4 + n])
            elif t == 2:
                d = unpack_strings(b[k + 4:k + 4 + n])
                # print d
            c = len(d)
            for j in range(i, i + c):
                newdata[j // ncols][j % ncols] = d[j - i]
            i += c
            k += 4 + n
        self.parent.update(newdata, x[0], x[1])


class quik_dde_server:
    # организует DDE сервер для получения данных из Quik

    def __init__(self, topic_name):
        self.data = None
        self.topic_name = topic_name

    def update(self, newdata, i0, j0):
        # свежие данные из Quik в переменной newdata, i0,j0 - координаты новых данных
        nrows = len(newdata)
        ncols = len(newdata[0])
        if self.data is None or len(self.data) < nrows or len(self.data[0]) < ncols:
            self.data = newdata
        elif i0<=len(self.data):
            for i in range(nrows):
                k = i0 - 1 + i
                if len(self.data[k]) == ncols:
                    self.data[k] = newdata[i]
                else:
                    print "Huyak"
                    for j in range(ncols):
                        self.data[k][j0 - 1 + j] = newdata[i][j]
        elif i0==len(self.data)+1:
            self.data+=newdata
        print len(self.data)

    def start(self):
        #запуск сервера
        server = dde.CreateServer()
        server.AddTopic(QuikTopic(self))
        server.Create(self.topic_name)
        event=win32event.CreateEvent(None,0,0,None)
        while 1:
            win32ui.PumpWaitingMessages(0,-1)
            rc=win32event.MsgWaitForMultipleObjects((event,),0,100,win32event.QS_ALLEVENTS)
            if rc==win32event.WAIT_OBJECT_0:
                break
            #elif rc==win32event.WAIT_OBJECT_0+1:
            #    print "OK1"
                #if win32ui.PumpWaitingMessages(0,-1):
                #    raise RuntimeError("We got an unexpected WM_QUIT message!")
            elif rc==win32event.WAIT_TIMEOUT:
                pass
                #print "Time-out elapsed"

class quik_dde_server_history(quik_dde_server):
    #dde server для получения истории изменения некоторого параметра в течении дня

    def __init__(self, topic_name):
        self.r=redis.Redis()
        quik_dde_server.__init__(self,topic_name)

    def path(self,date):
        #путь в redis к данным
        return self.topic_name+':'+date+':'

    def update(self, newdata, i0, j0):
        if len(newdata)==0:
            return
        date=newdata[0][-1]
        #date='06.05.2016'
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
