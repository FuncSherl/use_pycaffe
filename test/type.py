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
import math
  
import caffe 
import caffe.proto 
from caffe.proto import caffe_pb2  
import lmdb  
from matplotlib import pyplot as plt  

np.set_printoptions(threshold='nan')  #全部输出  

mnist_root=r'D:\workspaces\vs2013\caffe-master\examples\mnist\test_minist'
mnist_deploy=op.join(mnist_root, r'lenet.prototxt')
mnist_model=op.join(mnist_root, r'snapshot\lenet_iter_10000.caffemodel')
mnist_mean=op.join(mnist_root, r'minist_lmdb\mnist_mean.binaryproto')

mnist_test=op.join(mnist_root, r'minist_lmdb\mnist_test_lmdb')

#caffe.set_device(0)
#caffe.set_mode_gpu()
caffe.set_mode_cpu()

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
            #cv2.imwrite(op.join(r'lmdb_imgs',str(label)+r'.png'), data)
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
    #trans.set_mean('data', np.load(mean).mean(1).mean(1))
    trans.set_transpose('data', (2,0,1))
    
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
            
            #print net.blobs['data'].data
            out=net.forward()
            
            show_weights(net.params['conv1'])
            
            tep=net.blobs['conv1'].data[1]
            tep=tep[...]>128
            show_mid_resu(tep)
            
            
            #####
            tx=net.params['conv1'][0].data
            tx=np.transpose(tx,(1,0,2,3))
            print tx.shape
            tp=(tx.max()-tx.min())/2+tx.min()
            tx=tx[...]>tp
            
           
            show_mid_resu(  tx[0]  )
            ######
            
            
            #print net.blobs['prob'].data
            for j in range(cnt%batch+1):
                ret.append(net.blobs['prob'].data[j].flatten().argsort())  
                print net.blobs['prob'].data[j].flatten().argmax()
                
            if cnt>=len(dirls)-1:
                return ret
        cnt+=1
def show_weights(da):
    print da[0].data.shape
    print da[1].data.shape
    for i,d in enumerate(da[0].data):
        print d
        print da[1].data[i]

def show_mid_resu(data):
    l=math.sqrt(int(len(data)))
    fig = plt.figure()  
    
    for i,d in enumerate(data):
        ax = fig.add_subplot(l,l+1,i+1)  
        
        #d=d[...]>128
        print d
        ax.imshow(d, cmap="gray")  
    
    plt.show()              
           
def test_plt():
    img=cv2.imread(op.join(r'../lmdb_imgs', str(2)+r'.png'), cv2.IMREAD_GRAYSCALE)
    fig = plt.figure()  
    ax = fig.add_subplot(121)  
    ax.imshow(img)  
    ax.set_title("hei,i'am the first")  
  
    ax = fig.add_subplot(223)  
    ax.imshow(img)#以灰度图显示图片  , cmap="gray"
    ax.set_title("hei,i'am the second")#给图片加titile      
        
    plt.show()#显示刚才所画的所有操作  
    

if __name__ == '__main__':
    #read_lmdb(mnist_test)
    #test_plt()
    pa=con_mean(mnist_mean)
    
    paths=[]
    for i in range (10):
        paths.append(op.join(r'../lmdb_imgs', str(i)+r'.png'))
    print paths
    print recognize(paths, mnist_deploy, mnist_model, pa)
    pass