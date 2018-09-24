#coding:utf-8
'''
Created on Mar 15, 2018

@author: root
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import time
import os,cv2
import os.path as op

plt.ion()

caffe_root='/media/sherl/本地磁盘/FashionAI/algorithms/RMPE-master'
#img_dir = op.join(caffe_root,"data/MPII/images")
img_dir='/media/sherl/本地磁盘/FashionAI/train/Images/skirt'

import sys
sys.path.insert(0, op.join(caffe_root,'python'))
sys.path.insert(0,op.join(caffe_root,'examples/rmpe'))

from util.demo_pose_NMS import *
from util.cropBox import *

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

det_model_def = op.join(caffe_root,'models/VGG_SSD/deploy.prototxt')
det_model_weights = op.join(caffe_root,'models/VGG_SSD/VGG_MPII_COCO14_SSD_500x500_iter_60000.caffemodel')
det_net = caffe.Net(det_model_def,      # defines the structure of the model
                det_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

det_transformer = caffe.io.Transformer({'data': det_net.blobs['data'].data.shape})
det_transformer.set_transpose('data', (2, 0, 1))
det_transformer.set_mean('data', np.array([104,117,123])) # mean pixel
det_transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
det_transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

pose_model_def = op.join(caffe_root,'models/SPPE/deploy.prototxt')
pose_model_weights = op.join(caffe_root,'models/SPPE/shg+sstn.caffemodel')
pose_net = caffe.Net(pose_model_def,      # defines the structure of the model
                pose_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

pose_transformer = caffe.io.Transformer({'data': pose_net.blobs['data'].data.shape})
pose_transformer.set_transpose('data', (2, 0, 1))



image_resize = 500
det_net.blobs['data'].reshape(1,3,image_resize,image_resize)

configThred = 0.3#lower this threshold can improve recall but decrease precision, in our paper we use 0.09 阈

NMSThred = 0.45
#for img_name in open('examples/rmpe/util/test_images.txt','r'):  #Use this line to evaluate on the whole test test.
for img_name in os.listdir(img_dir):
    # check if image exists
    filename = op.join(img_dir, img_name.strip())
    if (os.path.isfile(filename) == False):
        print filename+" does not exist."
        continue
    image = caffe.io.load_image(filename)
        
    #Run the detection net and examine the top_k results
    transformed_image = det_transformer.preprocess('data', image)
    det_net.blobs['data'].data[...] = transformed_image
    # Forward pass.
    kepresu=det_net.forward()
    detections = kepresu['detection_out']
    
    print "blob params:"
    for i in det_net.blobs.keys():
        print i
    print "weight params :"
    for i in det_net.params.keys():
        print i
    
    print 'mbox_conf_flatten:',det_net.blobs['mbox_conf_flatten'].data.shape
    print 'mbox_loc:',det_net.blobs['mbox_loc'].data.shape
    print 'mbox_priorbox:',det_net.blobs['mbox_priorbox'].data.shape

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    top_indices = [m for m, conf in enumerate(det_conf) if conf > configThred]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()

    top_labels = det_label[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    ###########################################################
    mytest=cv2.imread(filename)
    mytest_resize=cv2.resize(mytest,(image_resize,image_resize))
    for ord, mi in enumerate(top_xmin):  
        colorm=[255,0,0]
        mytest_resize[int(top_ymin[ord]*image_resize):int(top_ymax[ord]*image_resize)-1, int(top_xmin[ord]*image_resize)]=colorm
        mytest_resize[int(top_ymin[ord]*image_resize):int(top_ymax[ord]*image_resize)-1, int(top_xmax[ord]*image_resize)]=colorm
        mytest_resize[int(top_ymin[ord]*image_resize),int(top_xmin[ord]*image_resize):int(top_xmax[ord]*image_resize)-1]=colorm
        mytest_resize[int(top_ymax[ord]*image_resize), int(top_xmin[ord]*image_resize):int(top_xmax[ord]*image_resize)-1]=colorm
        
        #mytest_resize[int(top_ymin[ord]*image_resize):int(top_ymax[ord]*image_resize), int(top_xmin[ord]*image_resize):int(top_xmax[ord]*image_resize)]=[255,0,0]
    cv2.imshow('vgg_ssd_out', mytest_resize)
    cv2.waitKey(0)

cv2.destroyAllWindows()


