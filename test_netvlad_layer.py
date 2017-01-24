## Testing the netVLAD layer. The funvtion is implemented in
## class VGGDescriptor in CartWheelFlow.py

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

# #
# # test-experiment tf.reshape
# np.random.seed(1)
# XX = np.zeros( (16,5,5,8) )
# for i in range(16):
#     for j in range(8):
#         XX[i,:,:,j] = np.floor(np.random.randn(5,5)*10) #np.ones((5,5,8)) * i
#
# tf_x = tf.constant( XX )
# shp = tf.shape(tf_x)
# tf_reshaped_x = tf.reshape( tf_x, [shp[0]*shp[1]*shp[2], shp[3]] )
# tf_sftmax = tf.nn.softmax(tf_reshaped_x, name='softmax_op' )
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# tff_x, tff_sftmax = sess.run( [tf_reshaped_x,tf_sftmax] )
#
# quit()


#
# # test-experiment : tf.segment_mean
# # make segments (batchwise)
# np.random.seed(1)
# e = []
# for ie in range(2):
#     e.append( np.ones(5, dtype='int32')*ie )
# e = np.hstack( e )
# tf_e = tf.constant( e, name='segment_e' )
#
# U = np.random.randint( 0, 10, (10,3) ).astype('float32')
# tf_U = tf.constant( U )
#
# seg_op = tf.segment_mean( tf_U, tf_e )
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# tff_seg = sess.run( seg_op )
# print tff_seg
#
# quit()


# # test-experiment : Slice and diag
# U = np.random.randint( 0, 10, (10,3) ).astype('float32')
# sm = tf.constant( U )
#
# Wsc = tf.unpack( sm, axis=1 )
# Wsc_diag = tf.diag( Wsc[0] )
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# tff_nj, tff_nj_diag = sess.run( [Wsc[0], Wsc_diag] )
#
#
# quit()

# #
# test-experiment netVLAD layer only
def verify_membership_mat( C ):
    print 'C.shape : ', C.shape # 16 x 60 x 80 x 64
    C_r = np.reshape( C, [16*60*80, 64 ] )

    #softmax computation for each row
    sm = np.zeros( C_r.shape )
    mmax = np.max( C_r, axis=1 )
    for i in range( sm.shape[0] ):
        sm[i] = np.exp( C_r[i,:] - mmax[i] ) / np.sum( np.exp( C_r[i,:] - mmax[i] ) )

    return sm


def verify_vlad( c, Xd, sm ):
    # c : cluster center 64x256
    # sm : 16*60*80 x 64
    # Xd : 16x60x80x256

    D = c.shape[1]
    K = c.shape[0]
    N = Xd.shape[0] * Xd.shape[1] * Xd.shape[2]
    b = Xd.shape[0]
    print 'verify_vlad : D=%d, K=%d, N=%d' %(D,K,N)

    X = Xd.reshape( (N, D) )
    vlad = []
    for k in range(K): #foreach cluster
        X_m = (X - c[k,:]) * sm[:,k:(k+1)]  * np.ones( (1,D) )
        #do segment_mean
        L = Xd.shape[1]*Xd.shape[2]

        vlad_k = []
        for bb in range(b):
            vlad_k.append( X_m[bb*L:(bb+1)*L].mean(axis=0) )
        vlad_k = np.vstack( vlad_k )
        vlad.append( vlad_k )
    vlad_2 = np.stack( vlad )

    veri_netvlad = vlad_2.transpose( (1,0,2) )
    #veri_netvlad is 16x64x256
    return veri_netvlad


    code.interact( local=locals() )


np.random.seed(1)
XX = np.zeros( (16,60,80,256) )
for i in range(16):
    for j in range(256):
        XX[i,:,:,j] = np.random.rand(60,80)#np.floor(np.random.rand(60,80)*10)  #np.ones((5,5,8)) * i



tf_x = tf.placeholder( 'float', [16,60,80,256], name='conv_desc' )
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor()
# tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
tf_sm, vlad_c, netvlad = vgg_obj.netvlad_layer( tf_x )

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tff_sm, tff_vlad_c, tff_netvlad = sess.run( [tf_sm, vlad_c, netvlad],  feed_dict={tf_x:XX})

# veri_sm = verify_membership_mat( tff_netvlad_conv )
veri_netvlad = verify_vlad( tff_vlad_c, XX, tff_sm )

quit()
