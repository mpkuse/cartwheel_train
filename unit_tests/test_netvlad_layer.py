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
# # test-experiment : log-sum-exp cost function testing
# np.random.seed(1)
# word = np.random.rand(16, 6*8*256 ).astype( 'float32')
# tf_vlad_word = tf.constant( word )
#
# vgg_obj = VGGDescriptor()
# vgg_obj.soft_ploss( tf_vlad_word, 5,10, 10.)
#
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# v = vgg_obj
#
# sp_q, sp_P, sp_N, tf_dis_q_P, tf_dis_q_N, rep_P,rep_N, pdis_diff, tff_cost = sess.run( [v.sp_q,v.sp_P,v.sp_N,   v.tf_dis_q_P,v.tf_dis_q_N,    v.rep_P,v.rep_N,   v.pdis_diff, v.cost] )
#
# quit()



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


## M is a 2D matrix
def zNormalize(M):
    return (M-M.mean())/(M.std()+0.0001)

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
        im_batch_normalized[b,:,:,0] = zNormalize( im_batch[b,:,:,0])
        im_batch_normalized[b,:,:,1] = zNormalize( im_batch[b,:,:,1])
        im_batch_normalized[b,:,:,2] = zNormalize( im_batch[b,:,:,2])
        # im_batch_normalized[b,:,:,:] = rgbnormalize( im_batch[b,:,:,:] )

    # cv2.imshow( 'org_', (im_batch[0,:,:,:]).astype('uint8') )
    # cv2.imshow( 'out_', (im_batch_normalized[0,:,:,:]*255.).astype('uint8') )
    # cv2.waitKey(0)
    # code.interact(local=locals())
    return im_batch_normalized


# test - looking at assignment of cluster. I have a doubt that the assignment is too soft (19th June, 2017)

# np.random.seed(1)
im_batch_size = 10
XX = np.zeros( (im_batch_size,240,320,3) )
for i in range(im_batch_size):
    nx = 600+2*i
    # nx = np.random.randint(1000)

    im_file_name = 'tf.logs/netvlad_k48/db_xl/im/%d.jpg' %(nx)
    # im_file_name = 'other_seqs/FAB_MAP_IJRR2008_DATA/City_Centre/Images/%04d.jpg' %(nx)
    print 'Reading : ', im_file_name
    XX[i,:,:,:] = cv2.resize( cv2.imread( im_file_name ), (320,240) )
y = normalize_batch( XX )





tf_x = tf.placeholder( 'float', [None,240,320,3], name='conv_desc' )
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor(b=im_batch_size, K=64)
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
# tf_vlad_word = vgg_obj.vgg16_raw_features(tf_x, is_training)
# netvlad = vgg_obj.netvlad_layer( tf_vlad_word )

sess = tf.Session()
if False: #random init
    sess.run(tf.global_variables_initializer())
else: # load from a trained model
    tensorflow_saver = tf.train.Saver()
    # PARAM_model_restore = 'tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
    # PARAM_model_restore = 'tf.logs/netvlad_k48/model-13000'
    # PARAM_model_restore = 'tf.logs/netvlad_k64_znormed/model-2000' # trained from 3d model z-normalize R,G,B individual,
    PARAM_model_restore = 'tf.logs/netvlad_k64_tokyoTM/model-2500'

    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver.restore( sess, PARAM_model_restore )



tff_netvlad, tff_sm = sess.run( [tf_vlad_word, vgg_obj.nl_sm ],  feed_dict={tf_x:y,  is_training:False, vgg_obj.initial_t: 0})

# veri_sm = verify_membership_mat( tff_netvlad_conv )
# veri_netvlad = verify_vlad( tff_vlad_c, XX, tff_sm )

# reshape tff_sm (b*60*80 x K)
tff_sm_variance = tff_sm[:,0]
for h in range(tff_sm.shape[0]):
    tff_sm_variance[h] = tff_sm[h,:].var()
tff_sm_variance = np.reshape( tff_sm_variance, [im_batch_size,60,80] )
#h=2800;plt.plot( tff_sm[h,:] ); plt.title( str(tff_sm[h,:].var()) ); plt.show()


Assgn_matrix = np.reshape( tff_sm, [im_batch_size,60,80,-1] ).argmax( axis=-1 ) #assuming batch size = 1

for i in range(im_batch_size):
    plt.subplot(2,2,1)
    plt.imshow( Assgn_matrix[i], cmap='hot')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow( XX[i,:,:,::-1].astype('uint8') )

    plt.subplot(2,2,3)
    plt.hist( Assgn_matrix[i].flatten(), bins=48 )

    plt.subplot(2,2,4)
    plt.imshow( -np.log(tff_sm_variance[i,:,:]), cmap='hot' )
    plt.colorbar()

    plt.show()



quit()
