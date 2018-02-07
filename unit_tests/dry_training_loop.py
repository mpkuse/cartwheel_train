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

## Given input to the vlad layer after wx+b (after conv), ie 16x60x80x64
## computes the softmax which denote cluster membership
def verify_membership_mat( C ):
    # print 'C.shape : ', C.shape # 16 x 60 x 80 x 64
    C_r = np.reshape( C, [16*60*80, 64 ] )

    #softmax computation for each row
    sm = np.zeros( C_r.shape )
    mmax = np.max( C_r, axis=1 )
    for i in range( sm.shape[0] ):
        sm[i] = np.exp( C_r[i,:] - mmax[i] ) / np.sum( np.exp( C_r[i,:] - mmax[i] ) )

    return sm

#note that `im_batch` is still a batch of color images need to make them to gray scale and then normalize
def normalize_batch_gray( im_batch ):
    #input : 16x240x320x3
    im_batch_gray = np.mean( im_batch, axis=3, keepdims=True ) / 255.0 #16x240x320x1
    # code.interact( local=locals() )
    return im_batch_gray


#TODO: parse_args for this var
PARAM_model_restore = None# 'tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'

#
# Tensorflow
tf_x = tf.placeholder( 'float', [16,240,320,1], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor()
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)

nP = 5
nN = 10
margin = 0.2#10.0
# fitting_loss = vgg_obj.svm_hinge_loss( tf_vlad_word, nP=nP, nN=nN, margin=margin )
# fitting_loss = vgg_obj.soft_ploss( tf_vlad_word, nP=nP, nN=nN, margin=margin )
fitting_loss = vgg_obj.soft_angular_ploss( tf_vlad_word, nP=nP, nN=nN, margin=margin )
pos_set_dev = vgg_obj.positive_set_std_dev( tf_vlad_word, nP=nP, nN=nN, scale_gamma=10. )

for vv in tf.trainable_variables():
    print 'name=', vv.name, 'shape=' ,vv.get_shape().as_list()
print '# of trainable_vars : ', len(tf.trainable_variables())


# restore saved model
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
if PARAM_model_restore == None:
    print tcolor.OKGREEN,'global_variables_initializer() : xavier', tcolor.ENDC
    tensorflow_session.run( tf.global_variables_initializer() )
else:
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


    #Remember to normalize images R=R/(R+G+B); G=G/(R+G+B) ; B=B/(R+G+B)

    im_batch_gray = normalize_batch_gray( im_batch ) #resulting in 16x240x320x1

    s = vgg_obj
    feed_dict = {tf_x : im_batch_gray,\
                 is_training:False,\
                 s.initial_t: 0,\
                }

    # proc = [tf_vlad_word, fitting_loss, pos_set_dev, s.p_sp_P, s.p_sp_N, s.p_XXt, s.p_masked_XXt, s.p_stddev, s.p_XYt, s.p_masked_XYt]
    # tff_vlad_word, tff_cost, tff_pos_set_dev, p_sp_P, p_sp_N, p_XXt, p_masked_XXt, p_stddev, p_XYt, p_masked_XYt  = tensorflow_session.run( proc, feed_dict=feed_dict )

    tff_cost, tff_vlad_word = tensorflow_session.run( [fitting_loss,tf_vlad_word], feed_dict=feed_dict)
    print 'cost : ', tff_cost

    # kk = 23
    # xxx = np.multiply( nl_sm[:,kk:kk+1] * np.ones((1,256)), nl_Xd - nl_c[kk,:] )
    # # xxx = nl_Xd - nl_c[kk,:]
    # xxx_s = xxx[0:4800,:].sum( axis=0 )
    # print 'max-err:', (xxx_s - nl_outputs[0,kk,:]).max()



    # tff_cost, tff_dis_q_P, tff_dis_q_N, pdis_diff = tensorflow_session.run( [fitting_loss, vgg_obj.tf_dis_q_P, vgg_obj.tf_dis_q_N, vgg_obj.pdis_diff], feed_dict=feed_dict )
    # np.set_printoptions( precision=3 )
    # print 'tff_cost:', tff_cost
    # print 'tff_dis_q_P:', tff_dis_q_P
    # print 'tff_dis_q_N:',tff_dis_q_N
    # print 'pdis_diff:', pdis_diff
    # code.interact( local=locals() )


    l += 1
