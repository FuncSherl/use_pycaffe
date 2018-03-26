#coding:utf-8
'''
Created on 2018年3月4日

@author:China
'''
import cv2, time
import os, random
import os.path as op
import numpy as np
import cut_video
from multiprocessing import Pool


def cut_video(inpu, output,time_len=10,num_def=1):   
    videoCapture = cv2.VideoCapture(inpu)  # 从文件读取视频  
    # 判断视频是否打开  
    if (videoCapture.isOpened()):  
        print 'Open:', inpu 
    else:  
        print 'Fail to open:', input  
    
    videorate = videoCapture.get(cv2.CAP_PROP_FPS)
    allcnt = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数

    
    # videoCapture.set(cv2.CAP_PROP_POS_AVI_RATIO,0.5)#CAP_PROP_POS_AVI_RATIO   cv2.CAP_PROP_FRAME_COUNT设置视频位置
    
    for i in range(num_def):
        if num_def > 1:
            tepoutput = op.splitext(output)[0] + "_" + str(i) + '.avi'  # 这样处理的视频结果是avi格式，只能以avi为后缀
        else:
            tepoutput = op.splitext(output)[0] + '.avi'  # 这样处理的视频结果是avi格式，只能以avi为后缀
        
        print "process:", tepoutput
        # 创建视频写入对象 
        video_writer = cv2.VideoWriter(tepoutput,
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    videorate,
                    (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        cnt = time_len * videorate  # 需要读取多少帧
    
        # print allcnt
        
        startfra = int(allcnt / (num_def + 1) * (i + 1) - cnt / 2)
        
        if startfra < 0:  # 错误处理
            startfra = 0
        elif startfra >= allcnt: 
            startfra = allcnt - 1
        
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, startfra)
        success, frame = videoCapture.read()  # 读取第一帧  
      
        while success and cnt > 0:  
            
            # frame = frame[0:1536,1200:1800]#截取画面  
            video_writer.write(frame)  # 将截取到的画面写入“新视频”  
            
            success, frame = videoCapture.read()  # 循环读取下一帧  
            # print success
            cnt -= 1
    videoCapture.release() 
    
    
def slice_video(inpu, output,time_len=10): 
    
    outcnt=0
    videoCapture = cv2.VideoCapture(inpu)  # 从文件读取视频  
    # 判断视频是否打开  
    if (videoCapture.isOpened()):  
        print 'Open:', inpu 
    else:  
        print 'Fail to open:', input 
        return 0 
    
    videorate = videoCapture.get(cv2.CAP_PROP_FPS)
    allcnt = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    # videoCapture.set(cv2.CAP_PROP_POS_AVI_RATIO,0.5)#CAP_PROP_POS_AVI_RATIO   cv2.CAP_PROP_FRAME_COUNT设置视频位置
    print videorate,"allcnt:",allcnt
    
    cnt = int(time_len * videorate)  # 需要读取多少帧

    # print allcnt
        
    #videoCapture.set(cv2.CAP_PROP_POS_FRAMES, startfra)
    
    frame_cnt=0
    success, frame = videoCapture.read()  # 读取第一帧  
      
    while success:  
        if frame_cnt%cnt==0:
            if allcnt-frame_cnt<cnt:
                print "left not enough"
                break
            outcnt+=1
            #cut_video.outcnt+=1
            tepoutput = op.splitext(output)[0] + "_"+str(int(frame_cnt/cnt+1)) + '.avi'  # 这样处理的视频结果是avi格式，只能以avi为后缀        
            print "process:", tepoutput
            # 创建视频写入对象 
            video_writer = cv2.VideoWriter(tepoutput,
                            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            videorate,
                            (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
        # frame = frame[0:1536,1200:1800]#截取画面  
        video_writer.write(frame)  # 将截取到的画面写入“新视频”  
        
        frame_cnt+=1
        success, frame = videoCapture.read()  # 循环读取下一帧  

    videoCapture.release() 
    return outcnt


def strip_video(inpu, output,rate_st=5,rate_ed=5): #前百分之多少、后百分之多少
    
    videoCapture = cv2.VideoCapture(inpu)  # 从文件读取视频  
    # 判断视频是否打开  
    if (videoCapture.isOpened()):  
        print 'Open:', inpu 
    else:  
        print 'Fail to open:', input  
    #output=op.splitext(output)[0]+".avi"
    videorate = videoCapture.get(cv2.CAP_PROP_FPS)
    allcnt = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数  
    # videoCapture.set(cv2.CAP_PROP_POS_AVI_RATIO,0.5)#CAP_PROP_POS_AVI_RATIO   cv2.CAP_PROP_FRAME_COUNT设置视频位置
    
    
    print "process:", inpu
    print cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),int(videoCapture.get(cv2.CAP_PROP_FOURCC))
        # 创建视频写入对象 
    video_writer = cv2.VideoWriter(output,
                    int(videoCapture.get(cv2.CAP_PROP_FOURCC)),#
                    videorate,
                    (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
    cnt = int((100-rate_st-rate_ed)*allcnt/100)  # 需要读取多少帧
        
    startfra=int(rate_st*allcnt/100)
        
    if startfra < 0:  # 错误处理
        startfra = 0
        
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, startfra)
    success, frame = videoCapture.read()  # 读取第一帧  
      
    while success and cnt > 0:  
            
        # frame = frame[0:1536,1200:1800]#截取画面  
        video_writer.write(frame)  # 将截取到的画面写入“新视频”  
            
        success, frame = videoCapture.read()  # 循环读取下一帧  
        # print success
        cnt -= 1
    videoCapture.release() 
    print "process:", inpu," done!!! -> ",output



if __name__ == '__main__':
    pass
