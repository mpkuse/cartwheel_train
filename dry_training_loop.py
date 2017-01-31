""" Dry Training Loop

    This script loads the learned weights. Uses the NetVLAD Renderer (which can give out a training tuple (q, [P]i, [N]j))
    For analysis/debug

    Created : 29th Jan, 2017
    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import code
import argparse

import tensorflow as tf
import tensorflow.contrib.slim as slim

from PandaRender import NetVLADRenderer
from CartWheelFlow import VGGDescriptor

#
import TerminalColors
tcolor = TerminalColors.bcolors()

#TODO: parse_args for this var
PARAM_model_restore = 'tf.logs/netvlad_logsumexp_loss/model-17500'

#
# Tensorflow
tf_x = tf.placeholder( 'float', [16,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor()
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)

nP = 5
nN = 10
margin = 10.0
# fitting_loss = vgg_obj.svm_hinge_loss( tf_vlad_word, nP=nP, nN=nN, margin=margin )
fitting_loss = vgg_obj.soft_ploss( tf_vlad_word, nP=nP, nN=nN, margin=margin )

for vv in tf.trainable_variables():
    print 'name=', vv.name, 'shape=' ,vv.get_shape().as_list()
print '# of trainable_vars : ', len(tf.trainable_variables())


# restore saved model
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_model_restore )


#
# Renderer (q, [P]. [N])
app = NetVLADRenderer()


l = 0
while True:
    im_batch, label_batch = app.step(16)
    while im_batch == None: #if queue not sufficiently filled, try again
        im_batch, label_batch = app.step(16)


    feed_dict = {tf_x : im_batch,\
                 is_training:False
                }

    tff_cost, tff_dis_q_P, tff_dis_q_N, pdis_diff = tensorflow_session.run( [fitting_loss, vgg_obj.tf_dis_q_P, vgg_obj.tf_dis_q_N, vgg_obj.pdis_diff], feed_dict=feed_dict )
    np.set_printoptions( precision=3 )
    print tff_cost
    print tff_dis_q_P
    print tff_dis_q_N
    print pdis_diff
    code.interact( local=locals() )


    l += 1
