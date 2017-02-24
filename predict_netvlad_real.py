""" Given a real camera image (captured from high flying drone, facing down). Does
    an ANN query on the KD-tree to find the most similar image. It uses the VLAD-like
    image representation

    Created : 25th Jan, 2017
    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
"""


import cv2
import numpy as np
import time
import argparse
import os
import pickle
import code
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim

from PandaRender import TrainRenderer
from CartWheelFlow import VGGDescriptor

from annoy import AnnoyIndex
import glob

#
import TerminalColors
tcolor = TerminalColors.bcolors()



PARAM_BAG_DUMP = './bag_dump/bag9/dji_sdk_'
PARAM_START = 1
PARAM_END   = len( glob.glob(PARAM_BAG_DUMP+'*.npz'))-2 #500
PARAM_MODEL = 'tf.logs/netvlad_hinged_logsumexploss_intranorm/model-4000'
PARAM_DB_PREFIX = 'tf.logs/netvlad_hinged_logsumexploss_intranorm/db1/'



def rgbnormalize( im ):
    im_R = im[:,:,0].astype('float32')
    im_G = im[:,:,1].astype('float32')
    im_B = im[:,:,2].astype('float32')
    S = im_R + im_G + im_B
    out_im = np.zeros(im.shape)
    out_im[:,:,0] = im_R / (S+1.0)
    out_im[:,:,1] = im_G / (S+1.0)
    out_im[:,:,2] = im_B / (S+1.0)

    return out_im


def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        im_batch_normalized[b,:,:,:] = rgbnormalize( im_batch[b,:,:,:] )
    return im_batch_normalized

#
# Init Tensorflow prediction
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor(b=1)
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)


#
# Load stored Weights
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_MODEL )


#
# Load ANN Index
with open( PARAM_DB_PREFIX+'vlad_word.pickle', 'r' ) as handle:
    print 'Read : ', PARAM_DB_PREFIX+'vlad_word.pickle'
    words_db = pickle.load( handle )

t_ann = AnnoyIndex( words_db.shape[1], metric='euclidean'  )

for i in range( words_db.shape[0] ):
    t_ann.add_item( i, words_db[i,:] )

print 'Rebuild ANN Index' #TODO: Figure out why t_ann.load() does not work
t_ann.build(10)


for ind in range( PARAM_START, PARAM_END ):
    npzFileName = PARAM_BAG_DUMP+str(ind)+'.npz'

    print tcolor.OKGREEN, 'Load NPZFile : ', npzFileName, 'of ', PARAM_END, tcolor.ENDC
    data = np.load( npzFileName )
    A = cv2.flip( data['A'], 0 )
    texPoses = data['texPoses']

    im_batch = np.zeros( (1,A.shape[0],A.shape[1],A.shape[2]) )
    im_batch[0,:,:,:] = A.astype('float32')

    feed_dict = {tf_x : im_batch,\
                 is_training:False,\
                 vgg_obj.initial_t: 0
                }

    startTime = time.time()
    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict=feed_dict )
    nn_indx, nn_dist = t_ann.get_nns_by_vector( tff_vlad_word[0,:], 10, include_distances=True )
    print 'predict_time = %5.2f ms' %( (time.time() - startTime)*1000. )

    print 'nn_dist : ', np.round(nn_dist,2)
    print 'nn_indx : ', nn_indx
    for en,h in enumerate(nn_indx):
        im_name = PARAM_DB_PREFIX+'/im/'+str(h)+'.jpg'
        # print 'Read Image : ', im_name
        q_im = cv2.imread( im_name )
        cv2.imshow( str(en), q_im )
        en_r = int(en / 5)
        en_c = en % 5
        cv2.moveWindow(str(en), 20+350*en_c, 20+350*en_r)

    cv2.imshow( 'win', cv2.cvtColor( A.astype('uint8'), cv2.COLOR_RGB2BGR ) )
    cv2.moveWindow('win', 20, 20+350+350*en_r)
    key = cv2.waitKey(0)
    if key == 27: #ESC
        break
