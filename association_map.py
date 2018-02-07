""" Loads a specified (learned) model. Loads the specified image.
    Computes the netvlad descriptor and the association map.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 30th Jan, 2018
"""

import cv2
import numpy as np
import os
import time
import code
import argparse
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

TF_MAJOR_VERSION = int(tf.__version__.split('.')[0])
TF_MINOR_VERSION = int(tf.__version__.split('.')[1])


from CartWheelFlow import VGGDescriptor
from ColorLUT import ColorLUT

import TerminalColors
tcolor = TerminalColors.bcolors()

## M is a 2D matrix
def zNormalize(M):
    return (M-M.mean())/(M.std()+0.0001)

def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        for ch in range(im_batch.shape[3]):
                im_batch_normalized[b,:,:,ch] = zNormalize( im_batch[b,:,:,ch])

    return im_batch_normalized

## Load Image
INPUT_FILE_NAME = 'sample_images/a0.jpg'

print 'Load Image : ', INPUT_FILE_NAME
IM = cv2.resize( cv2.imread( INPUT_FILE_NAME), (320, 240) )

## Network Params
NET_TYPE = "resnet6"
PARAM_K = 16
PARAM_model_restore = 'tf3.logs/B/model-8000'
# note, the model needs to be consistent with NET_TYPE, PARAM_K. 

####################### NOTHING TO EDIT BEYONG THIS POINT ##########################################
## Create Network
tf_x = tf.placeholder( 'float', [1,240,320,3], name='x' ) #this has to be 3 if training with color images
is_training = tf.placeholder( tf.bool, [], name='is_training')
vgg_obj = VGGDescriptor(K=PARAM_K, D=256, N=60*80, b=1)
tf_vlad_word = vgg_obj.network(tf_x, is_training, net_type=NET_TYPE )

# for vv in tf.trainable_variables():
    # print 'name=', vv.name, 'shape=' ,vv.get_shape().as_list()
# print '# of trainable_vars : ', len(tf.trainable_variables())

## Restore Model
sess = tf.Session()

print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
tensorflow_saver = tf.train.Saver()
tensorflow_saver.restore( sess, PARAM_model_restore )

## sess.run
im_batch = np.expand_dims( IM.astype('float32'), 0 )
im_batch_normalized = normalize_batch( im_batch )

feed_dict = {tf_x : im_batch_normalized,\
             is_training:True,\
             vgg_obj.initial_t: 0
            }

print 'Computing NetVLAD of input image'
tff_vlad_word, tff_sm = sess.run( [tf_vlad_word, vgg_obj.nl_sm], feed_dict=feed_dict)
Assgn_matrix = np.reshape( tff_sm, [1,60,80,-1] ).argmax( axis=-1 ) #assuming batch size = 1
print 'tff_vlad_word.shape', tff_vlad_word.shape

colorLUT = ColorLUT()
lut = colorLUT.lut( Assgn_matrix[0,:,:] )
cv2.imshow( 'IM', IM )
cv2.imshow( 'association map', cv2.resize( lut, (320,240) ) )
cv2.waitKey(0)
