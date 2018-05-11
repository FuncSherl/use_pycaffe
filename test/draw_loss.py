#coding:utf-8
'''
Created on Apr 25, 2018

@author: root
'''
import re
import matplotlib.pyplot as plt  
from time import  sleep

inputfile=r'/media/sherl/本地磁盘/wokmaterial/shz/res-student/changelossweight_log0503_1.txt'#r'/media/sherl/本地磁盘/wokmaterial/shz/res-student/onlytrain2048_log0427_3.txt'#r'/media/sherl/本地磁盘/wokmaterial/shz/log.txt'#

#
lossrestr=r'Train net output #1: prob2048\s*=\s*(.*?)\s*\('#r',\s*loss\s*=\s*(.*)'#
lossprob=r'Train net output #0: prob\s*=\s*(.*?)\s*\('
#Iteration 5160 (1.01918 iter/s, 19.6236s/20 iters), loss = 0.321512

lrrestr=r', lr =\s*(.*)'
lr_mult=1

accu=r'Test net output #0: acc/top-1 =\s*(.*)'

def findxy(idir):
    retloss=[]
    losssprob=[]
    retlr=[]
    retaccu=[]
    
    with open(idir, 'r') as f:
        for i in f.readlines():
            tep=re.search(lossrestr, i)
            if tep:
                nu=tep.group(1).strip()
                print nu
                retloss.append(float(nu))
                
            tep=re.search(lossprob, i)
            if tep:
                nu=tep.group(1).strip()
                print nu
                losssprob.append(float(nu))
            
            tep=re.search(lrrestr, i)
            if tep:
                nu=tep.group(1).strip()
                #print float(nu)*lr_mult
                retlr.append(float(nu)*lr_mult)
                
            tep=re.search(accu, i)
            if tep:
                nu=tep.group(1).strip()
                print 'accu:',nu
                retaccu.append(float(nu))
            
    return retloss,losssprob,retlr,retaccu



if __name__ == '__main__':
    plt.figure(1)
    while(1):
        tn,tpr,tlr,ac=findxy(inputfile)

        #plt.ion()
        plt.subplot(411)
        plt.plot(range(len(tn)),tn,color="blue")
        plt.subplot(413)
        plt.plot(range(len(tlr)),tlr,color="blue")
        plt.subplot(412)
        plt.plot(range(len(tpr)),tpr,color="blue")
        plt.subplot(414)
        plt.plot(range(len(ac)),ac,color="blue")
        #plt.show()
        plt.pause(5)
        plt.clf()
        print 'once'
    pass





