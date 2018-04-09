# coding:utf-8
'''
Created on 2018年3月2日

@author:China
'''
import cv2, time
import os, random
import os.path as op
import numpy as np
#import sendSMS as sms
from multiprocessing import Pool
import video_pro as vp

input_dir = r'/media/sherl/本地磁盘1/wokmaterial/ruizhi/BKVideos/validate/pos'#'E:/wokmaterial/ruizhi/normal_videos_unzip2'#
output_dir =r'/media/sherl/本地磁盘1/wokmaterial/ruizhi/BKVideos/validate/pos_imgs'# 'E:/wokmaterial/ruizhi/normal_videos_sliced10'#

time_len = 10  # how many seconds
num_def = 3  # 指定1个视频截取几个
report_gap=10#每切割多少个视频汇报一次

video_exts=['.mp4', '.avi', '.mkv','.flv','.3gp','.MP4','.MKV']

##############################################################################

procnt = 0  # 计数器
result=[]


def start(inpu, output):
    global procnt
    for i in os.listdir(inpu):
        tep_in = op.join(inpu, i)
        tep_out = op.join(output, i)

        if op.isdir(tep_in):
            if not op.exists(tep_out):
                os.makedirs(tep_out)
            start(tep_in, tep_out)
        else:
            ext = op.splitext(i)
            
            if ext[-1] in video_exts:
                vp.cut_video(tep_in, tep_out)
                procnt += 1
                if procnt and procnt % report_gap == 0:  # 处理进度汇报
                    #sms.sendlocalmes("13001071655", "cutvideo:" + str(procnt))
                    pass
            

if __name__ == '__main__':
    pro_pool = Pool()      
        
def multi_start(inpu, output):
    global procnt,result
    #print 'Run task %s (%s)...' % (inpu, os.getpid())

    for i in os.listdir(inpu):
        tep_in = op.join(inpu, i)
        tep_out = op.join(output, i)

        if op.isdir(tep_in):
            
            if not op.exists(tep_out):
                os.makedirs(tep_out)
            multi_start(tep_in, output)#output

        else:
            ext = op.splitext(i)
            
            if ext[-1] in video_exts:
                #pro_pool.apply_async(vp.cut_video, args=(tep_in, tep_out))
                # vp.cut_video(tep_in, tep_out)
                
                #result.append(pro_pool.apply_async(vp.slice_video, args=(tep_in, tep_out, time_len)))
                pro_pool.apply_async(vp.sanpshot_a_video, args=(tep_in, output,1))
                
                #vp.slice_video(tep_in, tep_out)

                procnt += 1
                '''
                if procnt and procnt % report_gap == 0:  # 处理进度汇报
                    outcnt=sum(map(lambda x:x.get(), result))
                    sms.sendlocalmes("13001071655", "cutvideo:" + str(procnt)+" outcnt:"+str(outcnt))
                '''
    #print 'Task %s runs at the end' % (inpu)
                    
            

    

if __name__ == '__main__':
    '''
    st = time.time()
    start(input_dir, output_dir)
    
    print "!!!!!!!!!!!!!!!!!!!!!!!!!start end: time :", (time.time() - st)
    '''
    st = time.time()
    print 'start multi_start at:',st
    
    multi_start(input_dir, output_dir)
    pro_pool.close()
    pro_pool.join()
    #outcnt=sum(map(lambda x:x.get(), result))
    print "!!!multi start end: time :", (time.time() - st),"cutvideo:done ",str(procnt)
    #sms.sendlocalmes("13001071655", "cutvideo:done "+str(procnt)+" times:"+str(time.time() - st))
    
    
    
