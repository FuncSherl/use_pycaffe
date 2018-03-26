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

img_dir = op.join(caffe_root,"data/MPII/images")

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
    detections = (det_net.forward()['detection_out'])

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
    ##################################################################
    mytest=cv2.imread(filename)
    mytest_resize=cv2.resize(mytest,(image_resize,image_resize))
    for ord, mi in enumerate(top_xmin):  
        colorm=[255,0,0]
        mytest_resize[int(top_ymin[ord]*image_resize):int(top_ymax[ord]*image_resize), int(top_xmin[ord]*image_resize)]=colorm
        mytest_resize[int(top_ymin[ord]*image_resize):int(top_ymax[ord]*image_resize), int(top_xmax[ord]*image_resize)]=colorm
        mytest_resize[int(top_ymin[ord]*image_resize),int(top_xmin[ord]*image_resize):int(top_xmax[ord]*image_resize)]=colorm
        mytest_resize[int(top_ymax[ord]*image_resize), int(top_xmin[ord]*image_resize):int(top_xmax[ord]*image_resize)]=colorm
        
        #mytest_resize[int(top_ymin[ord]*image_resize):int(top_ymax[ord]*image_resize), int(top_xmin[ord]*image_resize):int(top_xmax[ord]*image_resize)]=[255,0,0]
    cv2.imshow('vgg_ssd_out', mytest_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




#above is the VGG_SSD to identify the human pose

    # We scale the output bounding box of detection network to make sure we can crop the whole person
    scale_width = 1.3
    scale_height = 1.2

    preds_noNMS = []
    scores_noNMS = []
    bboxes = []

    for k in xrange(top_conf.shape[0]):
        label = top_labels[k]
        if (label != 1):
            continue
        xmin = int(round(top_xmin[k] * image.shape[1]))
        ymin = int(round(top_ymin[k] * image.shape[0]))
        xmax = int(round(top_xmax[k] * image.shape[1]))
        ymax = int(round(top_ymax[k] * image.shape[0]))
        
        # Get the coordinates for cropping
        img_height = np.size(image,0)
        img_width = np.size(image,1)
        width = xmax - xmin
        height = ymax - ymin
        xmin = int(max(0,xmin-width*(scale_width-1)/2))
        ymin = int(max(0,ymin-height*(scale_height-1)/2))
        xmax = int(min(img_width,xmax+width*(scale_width-1)/2))
        ymax = int(min(img_height,ymax+height*(scale_height-1)/2))
        
        cropped_image = cropBox(image,xmin,ymin,xmax,ymax)#!!!!!!!

        transformed_image = pose_transformer.preprocess('data', cropped_image) 
        pose_net.blobs['data'].data[...] = transformed_image
        # Forward pass.
        predictions = pose_net.forward()['prediction_heatmap']

        # Parse the outputs.
        pred_noNMS=[]
        score_noNMS=[]
        for i in range(0,16):
            real_loc = transformBoxInvert([predictions[0,0,:,3*i+0],predictions[0,0,:,3*i+1]],xmin,ymin,xmax,ymax,64)
            pred_noNMS.append(real_loc) #16 (x,y)
            score_noNMS.append(predictions[0,0,:,3*i+2])
        preds_noNMS.append(pred_noNMS)
        scores_noNMS.append(score_noNMS)
        bboxes.append([xmin,ymin,xmax,ymax])

    #run pose level NMS with threshold of number of match keypoints
    preds, scores = pose_NMS(preds_noNMS, scores_noNMS, bboxes)



    #pair keyppoint ids
    pairRef = [[1,2], [2,3],[3,7],
            [4,5], [4,7], [5,6], 
            [7,9], [9,10],
            [14,9],[11,12],[12,13],
            [13,9],[14,15],[15,16] ];
    partColor = [1,1,1,1,2,2,0,0,0,0,3,3,3,1,4,4];
    Colors = ['m', 'r', 'b', 'r', 'b']
    plt.gcf().clear()
    plt.imshow(image)
    #draw predicted pose
    for i in xrange(len(preds)):
        pred = preds[i]
        score = scores[i]
        for point_pair in pairRef:
            x1 = float(pred[point_pair[0]-1][0]); y1 = float(pred[point_pair[0]-1][1])
            x2 = float(pred[point_pair[1]-1][0]); y2 = float(pred[point_pair[1]-1][1])
            s1 = float(score[point_pair[0]-1]); s2 = float(score[point_pair[1]-1])
            if (s1 < 0.1 or s2 < 0.1):
                continue
            Color = Colors[partColor[point_pair[0]]]
            plt.plot([x1,x2], [y1,y2], color = Color, lw = 2)
            plt.axis('off')
    plt.draw()
    time.sleep(1)

if __name__ == '__main__':
    pass
    
    
    