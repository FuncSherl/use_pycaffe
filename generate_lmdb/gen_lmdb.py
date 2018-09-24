#coding:utf-8
'''
Created on 2018年3月9日

@author: sherl
'''

import os,random
import os.path as op

root=r'/media/sherl/WORK/workmaterials/imgs_1000'
train_txt=r'train_txt.txt'
test_txt=r'test_txt.txt'

def gen_list(ipath, train , test, classfile='classtodoc.txt', synset='synset_words.txt',rate=80):#rate% is train
    with open(synset,'r') as syn:
        tep_num=syn.readlines()
    
    label_num={}
    for i in tep_num:
        i=i.strip()
        label_num[i.split(' ')[0]]=' '.join(i.split(' ')[1:])
    
    
    classorder=1#类别要从1开始数起
    cnt_train=0
    cnt_test=0
    with open(train, 'w+') as tra:
        with open(test, 'w+') as tes:
            with open(classfile,'w+') as clasf:
            
                for i in os.listdir(ipath):   
                    i=i.strip()                
                    tep=op.join(ipath, i)
                    if op.isdir(tep):
                        clasf.write(str(classorder)+' '+label_num[i]+'\n')#记录类别与类序号对应
                        
                        for j in os.listdir(tep):#便利第二级目录，接触到图片
                            if op.splitext(j)[-1] in ['.jpg','.JPG','.PNG','.png']:
                                patho=op.join(i,j)
                                '''
                                if random.randint(1,100)<=96:
                                    continue
                                '''
                                print patho+' '+str(classorder)
                                if random.randint(1,100)<=rate:#随机定为train或者test集合里
                                    tra.write(patho+' '+str(classorder)+'\n')
                                    cnt_train+=1
                                else:
                                    tes.write(patho+' '+str(classorder)+'\n')
                                    cnt_test+=1
                                
                        
                        classorder+=1
    
    return [cnt_train, cnt_test]
        

                        
if __name__ == '__main__':
    print gen_list(root, train_txt, test_txt)
