"""
    Trying to implement a neural net-like computation-graph
    for point feature matching. This takes the CNN descriptor as input
    and produces point matching. Basic idea is to avoid computing
    daisy matching instead to rely on CNN dense feasures and info
    from netvlad.

    Loosely based on paper:
    "Rocco, Ignacio, Relja Arandjelovic, and Josef Sivic. "End-to-end weakly-supervised semantic alignment." arXiv preprint arXiv:1712.06861 (2017)."

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 20th March, 2018
"""

import cv2
import numpy as np
import os
import time
import code
import argparse
import sys
import pickle
import cv2
import code

# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# TF_MAJOR_VERSION = int(tf.__version__.split('.')[0])
# TF_MINOR_VERSION = int(tf.__version__.split('.')[1])
# from CartWheelFlow import VGGDescriptor

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

def tf_init():
    ## Network Params
    NET_TYPE = "resnet6"
    PARAM_K = 16
    PARAM_model_restore = 'tf3.logs/B/model-8000'


    ## Create Network
    tf_x = tf.placeholder( 'float', [1,240,320,3], name='x' ) #this has to be 3 if training with color images
    is_training = tf.placeholder( tf.bool, [], name='is_training')
    vgg_obj = VGGDescriptor(K=PARAM_K, D=256, N=60*80, b=1)
    tf_vlad_word = vgg_obj.network(tf_x, is_training, net_type=NET_TYPE )


    ## Sess + Load
    sess = tf.Session()

    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver = tf.train.Saver()
    tensorflow_saver.restore( sess, PARAM_model_restore )

    Q = {}
    Q['tf_x'] = tf_x
    Q['is_training'] = is_training
    Q['vgg_obj'] = vgg_obj
    Q['sess'] = sess
    return Q


DATA = pickle.load( open('DATA.pickle', 'rb') )
x0 = DATA['_cnn_normed'][0][0,:,:,:]
x1 = DATA['_cnn_normed'][1][0,:,:,:]

y0 = np.reshape( x0 , ( x0.shape[0]*x0.shape[1], -1 ) )
y1 = np.reshape( x1 , ( x1.shape[0]*x1.shape[1], -1 ) )

T = np.tensordot( x0, x1, [2,2] ) #if u do tensordot dont need to reshape etc, can also be directly accomplished with tensorflow
# T[10,20,:,:] will give all the nearests dots in 2nd image with 10,20 of 1st
code.interact( local=locals() )


for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        p = T[i,j,:,:]
        p = (255. * (p+1.)/2.).astype( 'uint8' )
        im_falsecolor = cv2.resize( cv2.applyColorMap( p, cv2.COLORMAP_HOT ), (320,240) )

        cv2.imshow( 'im_falsecolor', im_falsecolor )

        im0 = DATA['_im'][0].copy()
        im1 = DATA['_im'][1].copy()

        x_im0 = cv2.circle( im0, (4*j,4*i), 2, (0,255,0), -1 )
        cv2.imshow( 'x_im0', x_im0 )
        cv2.imshow( 'x_im1', (0.3*im1+0.7*im_falsecolor).astype('uint8') )
        cv2.waitKey(0)
quit()



######################################
Q = tf_init()

## Run
INPUT_FILE_NAME = 'sample_images/a0.jpg'

DATA = {}
DATA['_cnn'] = []
DATA['_cnn_normed'] = []
DATA['_vlad_word'] = []
DATA['_sm'] = []
DATA['_im'] = []
for INPUT_FILE_NAME in ['sample_images/b0.jpg', 'sample_images/b1.jpg']:
    print 'Read: ', INPUT_FILE_NAME
    IM = cv2.resize( cv2.imread( INPUT_FILE_NAME), (320, 240) )


    im_batch = np.expand_dims( IM.astype('float32'), 0 )
    im_batch_normalized = normalize_batch( im_batch )

    feed_dict = {Q['tf_x'] : im_batch_normalized,\
                 Q['is_training']:True,\
                 Q['vgg_obj'].initial_t: 0
                }

    # tff_vlad_word, tff_sm = sess.run( [tf_vlad_word, vgg_obj.nl_sm], feed_dict=feed_dict)

    nl_input_normalized = tf.nn.l2_normalize( Q['vgg_obj'].nl_input, dim=3 ) #dim is now deprecated, instead use axis
    tff_cnn_normed, tff_cnn, tff_vlad_word, tff_sm = Q['sess'].run(
                                [
                                nl_input_normalized,
                                Q['vgg_obj'].nl_input,
                                Q['vgg_obj'].nl_outputs,
                                Q['vgg_obj'].nl_sm
                                ], feed_dict=feed_dict)
    DATA['_im'].append( IM )
    DATA['_cnn'].append( tff_cnn )
    DATA['_cnn_normed'].append( tff_cnn_normed )
    DATA['_vlad_word'].append( tff_vlad_word )
    DATA['_sm'].append( tff_sm )
pickle.dump( DATA, open( 'DATA.pickle', 'wb') )
