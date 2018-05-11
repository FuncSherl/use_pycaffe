#coding:utf-8
'''
Created on Apr 26, 2018

@author: root
'''
import os,caffe
import os.path as op
#resnet50
rootdir=r'/media/sherl/本地磁盘/wokmaterial/ruizhi/nets/resnet_50_BK'
Bkmodel=op.join(rootdir, r'resnet50_snapshot/resnet50_iter_470000.caffemodel')
BKdeploy=op.join(rootdir, r'ResNet-50-deploy_forpython.prototxt')

net_resnet50=caffe.Net(BKdeploy, Bkmodel, caffe.TEST)

#resnet18
root18=r'/media/sherl/本地磁盘/wokmaterial/shz/res-student'
deploy18=op.join(root18, r'resnet_student_train.prototxt')
model18=op.join(root18, r'resnet_student_snapshot_64/resnet_student_iter_20000.caffemodel')

net_resnet18=caffe.Net(deploy18, model18, caffe.TEST)

print net_resnet18.params['fc2'][0].data.shape
print net_resnet50.params['fc2'][0].data.shape

net_resnet18.params['fc2'][0].data[...]=net_resnet50.params['fc2'][0].data[...]
net_resnet18.params['fc2'][1].data[...]=net_resnet50.params['fc2'][1].data[...]

net_resnet18.save(op.join(root18,r'finturn_fc2fromres50.caffemodel'))


if __name__ == '__main__':
    pass