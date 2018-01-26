""" Make ROC Plots

    Uses a trained model. Uses data from Pitssburg dataset to compute the
    NetVLAD descriptors. Having known the ground truth will compute confusion matrix.
    And then eventually the ROC plot for various thresholds.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 24th Jan, 2018
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import code
import argparse
import os
import pickle

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.tensorboard.plugins import projector #for t-SNE visualization

from TimeMachineRender import TimeMachineRender
from PittsburgRenderer import PittsburgRenderer


from CartWheelFlow import VGGDescriptor


#
import TerminalColors
tcolor = TerminalColors.bcolors()


#####
##### Normalization
#####
def print_trainable_vars():
    for vv in tf.trainable_variables():
        print tcolor.OKGREEN, 'name=', vv.name, 'shape=' ,vv.get_shape().as_list(), tcolor.ENDC
    print '# of trainable_vars : ', len(tf.trainable_variables())


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

## M is a 2D matrix
def zNormalize(M):
    return (M-M.mean())/(M.std()+0.0001)

def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        im_batch_normalized[b,:,:,0] = zNormalize( im_batch[b,:,:,0])
        im_batch_normalized[b,:,:,1] = zNormalize( im_batch[b,:,:,1])
        im_batch_normalized[b,:,:,2] = zNormalize( im_batch[b,:,:,2])
        # im_batch_normalized[b,:,:,:] = rgbnormalize( im_batch[b,:,:,:] )
    return im_batch_normalized


##### End Normalization


PARAM_NET_TYPE = "resnet6"
PARAM_MODEL = "tfsuper.logs/C/model-500"
PARAM_K = 16
nP = 9
nN = 10
THRESHOLD = 0.4 #In logistic space. ie. if logistic scores are greater than this number ==> +ve


batch_size = 1+nP+nN
ground_truth = np.append(   np.ones( 1+nP, dtype='bool'), np.zeros( nN, dtype='bool' )   )
yes_yes = 0
false_match_accepted = 0
true_match_rejected = 0
no_no = 0


####
#### Define tensorflow computation-graph
####
tf_x = tf.placeholder( 'float', [batch_size,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor(K=PARAM_K, D=256, N=60*80, b=batch_size)
print tcolor.OKGREEN,'Network : ', PARAM_NET_TYPE, tcolor.ENDC
tf_vlad_word = vgg_obj.network(tf_x, is_training, PARAM_NET_TYPE)



####
#### Load stored Weights
####
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_MODEL )



PTS_BASE = 'data_Akihiko_Torii/Pitssburg/'
print 'Start Renderer. Data Path: ', PTS_BASE
pr = PittsburgRenderer( PTS_BASE )
for i in range(20):
    a,_ = pr.step(nP=nP, nN=nN, apply_distortions=False)
    # a: 1+nP+nN x 240 x 320 x 3
    # a[0] : I_q
    # a[1:nP+1]: I_{P_i}
    # a[nP+1:nP+nN+1]: I_{N_i}

    print i,a.shape

    a_i = np.expand_dims(a[6,:,:,:], 0)


    # a_i_normed = normalize_batch( a_i )
    a_i_normed = normalize_batch( a )

    feed_dict = {tf_x : a_i_normed,\
                 is_training:False,\
                 vgg_obj.initial_t: 0}

    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict=feed_dict )
    print 'tff_vlad_word.shape : ', tff_vlad_word.shape

    # Score
    for k in [0]: #range(0,1+nP):
        DOT_word = np.dot( tff_vlad_word[k,:], np.transpose(tff_vlad_word[:,:]) )
        sim_scores = np.sqrt( 1.0 - np.minimum(1.0, DOT_word ) )
        sim_scores_logistic = (1.0 / (1.0 + np.exp( 11.0*sim_scores - 3.0 )) + 0.01)
        # print k, np.round(sim_scores_logistic, 3 )

        predictions = sim_scores_logistic > THRESHOLD
        # ground_truth
        for l in range( len(predictions) ):
            if (predictions[l] == True) and (ground_truth[l] == True):
                yes_yes = yes_yes + 1
            if (predictions[l] == True) and (ground_truth[l] == False):
                false_match_accepted = false_match_accepted + 1
            if (predictions[l] == False) and (ground_truth[l] == True):
                true_match_rejected = true_match_rejected + 1
            if (predictions[l] == False) and (ground_truth[l] == False):
                no_no = no_no + 1


    print 'yes_yes=%4d, false_match_accepted=%4d, true_match_rejected=%4d, no_no=%4d' %(yes_yes, false_match_accepted, true_match_rejected, no_no)
    # code.interact( local=locals() )

    # cv2.waitKey(0)
quit()
