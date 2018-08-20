"""
    A class interfce to netvlad based whole image descriptor. To use the
    pre-trained network in your application use this code and unit-test

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 20th Aug, 2018
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

class WholeImageDescriptor:
    def __init__( self, NET_TYPE, PARAM_K, PARAM_model_restore ):
        self.NET_TYPE = NET_TYPE
        self.PARAM_K = PARAM_K
        self.PARAM_model_restore = PARAM_model_restore

        ## Create Network
        tf_x = tf.placeholder( 'float', [1,240,320,3], name='x' ) #this has to be 3 if training with color images
        is_training = tf.placeholder( tf.bool, [], name='is_training')
        vgg_obj = VGGDescriptor(K=PARAM_K, D=256, N=60*80, b=1)
        tf_vlad_word = vgg_obj.network(tf_x, is_training, net_type=NET_TYPE )

        ## Restore Model
        sess = tf.Session()

        print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
        tensorflow_saver = tf.train.Saver()
        tensorflow_saver.restore( sess, PARAM_model_restore )

        self.tf_x = tf_x
        self.tf_vlad_word = tf_vlad_word
        self.is_training = is_training
        self.vgg_obj = vgg_obj
        self.sess = sess


    def get_descriptor( self, im ):
        """ im: 1x240x320x3 """
        assert( len(im.shape) == 4 )
        feed_dict = {self.tf_x : im,\
                     self.is_training:True,\
                     self.vgg_obj.initial_t: 0
                    }

        tff_vlad_word, tff_sm = self.sess.run( [self.tf_vlad_word, self.vgg_obj.nl_sm], feed_dict=feed_dict)
        Assgn_matrix = np.reshape( tff_sm, [1,60,80,-1] ).argmax( axis=-1 ) #assuming batch size = 1

        return tff_vlad_word, Assgn_matrix


if __name__=='__main__':
    ## Network Params
    NET_TYPE = "resnet6"
    PARAM_K = 16
    PARAM_model_restore = './tfmodels/B_vgg/model-8000'
    PARAM_model_restore = './tfmodels/D/model-8000'

    WID_net = WholeImageDescriptor( NET_TYPE, PARAM_K, PARAM_model_restore )


    ## Load Image
    INPUT_FILE_NAME = 'sample_images/a0.jpg'
    print 'Load Image : ', INPUT_FILE_NAME
    IM = cv2.resize( cv2.imread( INPUT_FILE_NAME), (320, 240) )
    im_batch = np.expand_dims( IM.astype('float32'), 0 )


    ## descriptor and association map
    ##      tff_vlad_word : 1x4096
    ##      Assgn_matrix  : 1x60x80
    tff_vlad_word, Assgn_matrix = WID_net.get_descriptor( im_batch )


    ## Visualize Assgn_matrix - as a false color map
    colorLUT = ColorLUT()
    lut = colorLUT.lut( Assgn_matrix[0,:,:] )
    cv2.imshow( 'IM', IM )
    cv2.imshow( 'Assgn_matrix', cv2.resize( lut, (320,240) ) )
    cv2.waitKey(0)
