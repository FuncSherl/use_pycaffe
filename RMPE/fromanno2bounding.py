#coding:utf-8
'''
Created on Mar 16, 2018

@author: root
'''
import os,csv
import os.path as op
import h5py

rootdir='/media/sherl/本地磁盘/FashionAI/keypoint'

annotatepath=op.join(rootdir, 'Annotations/annotations.csv')
imgdir=op.join(rootdir, 'Images')


outputdir='/media/sherl/本地磁盘/FashionAI/hdf5_train'
#Images/trousers/67a24d0b384d1355a71b80ece884c29a.jpg,trousers,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,-1_-1_-1,96_72_1,285_68_1,-1_-1_-1,-1_-1_-1,206_257_1,203_312_1,64_289_1,204_297_1,335_256_1
def pro_line(inpu):#process a annotation line return filedir,type, [xmin, ymin, xmax, ymax]
    xmin=ymin=100000
    xmax=ymax=0
    for i in inpu[2:]:#rid the first 2 col
        tep=map(int,i.split('_'))
        print tep
        if tep[-1]!=-1:
            if xmax<tep[0]:  xmax=tep[0]
            if xmin>tep[0]:  xmin=tep[0]
            
            if ymin>tep[1]:  ymin=tep[1]
            if ymax<tep[1]:  ymax=tep[1]
    return inpu[0],input[1], [xmin, ymin, xmax ,ymax]

def pro_annotate(inpu):#form a map with key =dir
    ret={}
    reader = csv.reader(file(inpu, 'rb'))
    for i in reader:
        dirs,typ,tep=pro_line(i)
        ret[dirs]=tep
    
    return ret
        
def gen_hdf5(hdf5_data_filename):
    '''
    HDF5 is one of the data formats Caffe accepts
    '''
    with h5py.File(hdf5_data_filename, 'w') as f:
        dir_label=pro_annotate(annotatepath)
        for i in os.listdir(imgdir):
            i=i.strip()
            outtep=op.join(outputdir, i)
            if not op.exists(outtep):
                os.mkdir(outtep)
            
            pwddir=op.join(imgdir, i)
            for j in os.listdir(pwddir):
                j=j.strip()
                mapkey='Images/'+i+'/'+j
                maplabel=dir_label[mapkey]
                
            pass
    
        #f['data'] = data['input'].astype(np.float32)
        #f['label'] = data['output'].astype(np.float32)
        

if __name__ == '__main__':
    pass