#coding:utf-8
'''
Created on 2017��11��12��

@author: sherl
'''

import os  
import os.path as op
import glob  
import random  
import numpy as np  
  
import cv2  
  
import caffe 
import caffe.proto 
from caffe.proto import caffe_pb2  
import lmdb  

mnist_root=r'F:\workspaces\vs2013\caffe-master\caffe-master\TEST_MINIST'
mnist_deploy=op.join(mnist_root, r'lenet.prototxt')
mnist_model=op.join(mnist_root, r'snapshot\minist_iter_10000.caffemodel')
mnist_mean=op.join(mnist_root, r'minist_lmdb\mnist_mean.binaryproto')

caffe.set_device(0)
caffe.set_mode_gpu()

def read_lmdb(lmdb_path):

    #open lmdb
    lmdb_env = lmdb.open(lmdb_path)
    #begin transaction
    lmdb_txn = lmdb_env.begin()
    #get cursor
    lmdb_cursor = lmdb_txn.cursor()
    #get data object
    datum = caffe.proto.caffe_pb2.Datum()

    for key, value in lmdb_cursor:
        #parse back to datum
        datum.ParseFromString(value)
        #get y value
        label = datum.label
        print('label = ' + str(label))
        data_array = caffe.io.datum_to_array(datum)
        
        
        
        print('data is numpy.ndarray :')
        for data in data_array:
            cv2.imshow('test',data)
            cv2.waitKey(0)

    lmdb_env.close()

def con_mean(path):
    blob = caffe.proto.caffe_pb2.BlobProto()           # 创建protobuf blob
    data = open(path, 'rb' ).read()         # 读入mean.binaryproto文件内容
    blob.ParseFromString(data)                         # 解析文件内容到blob

    array = np.array(caffe.io.blobproto_to_array(blob))# 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）
    mean_npy = array[0]                                # 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
    
    out=op.splitext(path)[0]+r'.npy'
    np.save(out ,mean_npy)
    return out

def recognize(dirls, deploy, model, mean):             #用opencv读入图片就不要channel_swap and raw_scale
    net=caffe.Net(deploy, model, caffe.TEST)
    trans=caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    trans.set_mean('data', np.load(mean).mean(1).mean(1))
    trans.set_transpose('data', (2,1,0))
    
    shap=net.blobs['data'].data.shape
    batch=shap[0]
    
    ret=[]
    
    cnt=0    
    
    for i in dirls:
        imt=cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        res=cv2.resize(imt,(shap[2],shap[3]),interpolation=cv2.INTER_CUBIC)
        
        #cv2.imshow('t', res)
        #cv2.waitKey(0)
        
        res=res[:,:,np.newaxis]
        
        net.blobs['data'].data[cnt%batch]=trans.preprocess('data', res)
        
            
        if cnt>=len(dirls)-1 or cnt%batch>=batch-1:
            
            print net.blobs['data'].data
            out=net.forward()
            
            print net.blobs['prob'].data
            for j in range(cnt%batch+1):
                ret.append(net.blobs['prob'].data[j].flatten().argsort())  
                print net.blobs['prob'].data[j].flatten().argmax()
                
            if cnt>=len(dirls)-1:
                return ret
        cnt+=1
            
            
        
        
    
    

if __name__ == '__main__':
    #read_lmdb(r'F:\workspaces\vs2013\caffe-master\caffe-master\TEST_MINIST\minist_lmdb\mnist_test_lmdb')
    pa=con_mean(mnist_mean)
    
    paths=[]
    for i in range (10):
        paths.append(op.join(r'../handnums', str(i)+r'.bmp'))
        paths.append(op.join(r'../handnums', str(i*11)+r'.bmp'))
    print paths
    print recognize(paths, mnist_deploy, mnist_model, pa)
    pass