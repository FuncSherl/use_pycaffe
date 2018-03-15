#coding:utf-8
'''
Created on Mar 14, 2018

@author: sherl
'''
import os
import os.path as op
import random

def genlist(indir, trainfile='./train.txt', testfile='./test.txt', mapfile='./maplable.txt',rate=85):
    with open(trainfile, 'w+') as tra:
        with open(testfile, 'w+') as tes:
            with open(mapfile, 'w+') as mapf:
                cnt=0
                cnt_tra=0
                cnt_test=0
                for i in os.listdir(indir):
                    cnt+=1
                    tepdir=op.join(indir, i)
                    mapf.write(i+' '+str(cnt)+'\n' )
                    for j in os.listdir(tepdir):
                        kep=op.join(i,j)
                        print kep
                        if random.randint(1,100)<rate:
                            tra.write(kep+' '+str(cnt)+'\n')
                            cnt_tra+=1
                        else:
                            tes.write(kep+' '+str(cnt)+'\n')
                            cnt_test+=1
                             
    
    print cnt,cnt_tra, cnt_test                    

if __name__ == '__main__':
    genlist('/media/sherl/本地磁盘1/data_DL/unzipFile')