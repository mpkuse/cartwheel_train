""" Convert DB Styles

    Created : 17th Feb, 2017
    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
"""
import cv2
import numpy as np
import code
import glob
import time

INPUT_PREFIX = 'other_seqs/Lip6OutdoorDataSet/Images/outdoor_kennedylong_'
INPUT_PREFIX = 'other_seqs/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_'
INPUT_PREFIX = 'other_seqs/data_collection_20100901/imgs/img_'
INPUT_PREFIX = 'other_seqs/kitti_dataset/sequences/06/image_2/00'

OUTPUT_PREFIX = 'other_seqs/kitti_dataset_npz/06/'

PARAM_START = 0
PARAM_END   =  len( glob.glob(INPUT_PREFIX+'*.png'))-1

for i in range(PARAM_START, PARAM_END):
    startTime = time.time()
    print '---%d of %d---' %(i,PARAM_END)
    seq = '%s%04d.png' %(INPUT_PREFIX,i)

    # inputFName = glob.glob( seq )
    # print len(inputFName), seq, inputFName
    # inputFName = inputFName[0] #'%s%06d.ppm' %(INPUT_PREFIX, i)
    inputFName = seq

    outputFName = '%s%d.npz' %(OUTPUT_PREFIX, i)

    im_raw = cv2.imread( inputFName )
    print 'READ  : ',inputFName

    cv2.imshow( 'win_raw', im_raw )
    cv2.waitKey(10)

    if OUTPUT_PREFIX is None:
        continue

    # reshape im
    im_ = cv2.resize( im_raw, (320,240) )
    im__ = cv2.cvtColor( im_.astype('uint8'), cv2.COLOR_RGB2BGR )

    np.savez( outputFName, A=im__ )
    cv2.imwrite( outputFName+'.png', im_ )
    print 'WRITE : ', outputFName

    cv2.imshow( 'win_proc', im_ )
    cv2.waitKey(10)

    print 'Done in %4.2fms' %((time.time() - startTime)*1000.)
