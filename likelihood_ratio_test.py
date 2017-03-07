""" Likelihood Ratio Test of the NetVLAD descriptors
        Computes the Likelihood ratio test of the trained descriptors' ability.
        LR = frac{ Probab( dot(X,Y) > 0.8 / same place )} { Probab( dot(X,Y) > 0.8 / diff place) }

        This can be realized by drawing random samples from PandaRenderer/NetVLADrenderer

        Created : 22nd Feb, 2017
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
# from PandaRender import TrainRenderer
from CartWheelFlow import VGGDescriptor
import DimRed

#
import TerminalColors
tcolor = TerminalColors.bcolors()

PARAM_MODEL = 'tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
PARAM_MODEL_DIM_RED = 'tf.logs/siamese_dimred_fc/model-400'
PARAM_N_RENDERS = 200 #Number of renders in a run
PARAM_N_RUNS    = 20  #Number of runs
PARAM_THRESH = np.arange( 0.01, 1.99, 0.01 ) #[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65 ]

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

    # cv2.imshow( 'org_', (im_batch[0,:,:,:]).astype('uint8') )
    # cv2.imshow( 'out_', (im_batch_normalized[0,:,:,:]*255.).astype('uint8') )
    # cv2.waitKey(0)
    # code.interact(local=locals())
    return im_batch_normalized

def thumbnail_batch( im_batch ):
    dim = (im_batch.shape[0],int(im_batch.shape[1]/5),int(im_batch.shape[2]/5),im_batch.shape[3])
    thumbs = np.zeros(dim, dtype='uint8')
    for b in range(im_batch.shape[0]):
        thumbs[b,:,:,:] = cv2.resize( cv2.cvtColor(im_batch[b,:,:,:].astype('uint8'), cv2.COLOR_RGB2BGR ), (0,0), fx=0.2, fy=0.2 )
    return thumbs

## tff_vlad_word is 16x8192. 8192 is actually not fixed and 2nd dim can be anything. Also assumes
## the 16 words are organized as follows <query, 5 postives, 10 negatives>
def get_confusion_mat(tff_vlad_word, THRESH):
    pred = (1.0 - np.dot( tff_vlad_word[0,:], np.transpose( tff_vlad_word[1:,:] ) ) ) < THRESH
    grtr = np.hstack( (np.ones( 5, dtype=bool ), np.zeros( 10, dtype=bool ) ) ) #5xTrue 10xFalse
    C =  np.zeros( (2,2) )
    C[0,0] = ((grtr == True) & (pred == True)).sum()
    C[1,1] = ((grtr == False) & (pred == False)).sum()
    C[0,1] = ((grtr == True) & (pred == False)).sum()
    C[1,0] = ((grtr == False) & (pred == True)).sum()
    return C

## More probability analysis on confusion matrix
def conf_mat_analysis( C, verbose=False ):
    t_events = C.sum() #total events
    Pr_same = (C[0,0] + C[0,1]) / t_events
    Pr_diff = (C[1,0] + C[1,1]) / t_events

    Pr_pred_same = (C[0,0] + C[1,0])/t_events
    Pr_pred_diff = (C[0,1] + C[1,1])/t_events


    C_joint = C / t_events
    ps_g_gs = C_joint[0,0]/Pr_same #Pr(pred to b same/same)
    pd_g_gs = C_joint[0,1]/Pr_same
    ps_g_gd = C_joint[1,0]/Pr_diff
    pd_g_gd = C_joint[1,1]/Pr_diff

    gs_g_ps = C_joint[0,0]/Pr_pred_same
    gs_g_pd = C_joint[0,1]/Pr_pred_diff
    gd_g_ps = C_joint[1,0]/Pr_pred_same
    gd_g_pd = C_joint[1,1]/Pr_pred_diff

    if verbose == True:
        print 'Pr(same place) = %6.4f' %( Pr_same )
        print 'Pr(diff place) = %6.4f' %( Pr_diff )
        print 'Pr(pred to be same) = %6.4f' %(Pr_pred_same)
        print 'Pr(pred to be diff) = %6.4f' %(Pr_pred_diff)

        print 'Pr(pred to b same/same) = %6.4f' %(C_joint[0,0]/Pr_same )
        print 'Pr(pred to b diff/same) = %6.4f' %(C_joint[0,1]/Pr_same )
        print 'Pr(pred to b same/diff) = %6.4f' %(C_joint[1,0]/Pr_diff )
        print 'Pr(pred to b diff/diff) = %6.4f' %(C_joint[1,1]/Pr_diff )

        print 'Pr(same/pred to b same) = %6.4f' %(C_joint[0,0]/Pr_pred_same )
        print 'Pr(same/pred to b diff) = %6.4f' %(C_joint[0,1]/Pr_pred_diff )
        print 'Pr(diff/pred to b same) = %6.4f' %(C_joint[1,0]/Pr_pred_same )
        print 'Pr(diff/pred to b diff) = %6.4f' %(C_joint[1,1]/Pr_pred_diff )

        print 'Pr(same,pred to b same) = %6.4f' %(C_joint[0,0])
        print 'Pr(same,pred to b diff) = %6.4f' %(C_joint[0,1])
        print 'Pr(diff,pred to b same) = %6.4f' %(C_joint[1,0])
        print 'Pr(diff,pred to b diff) = %6.4f' %(C_joint[1,1])

    # return s_g_s, d_g_s, s_g_d, d_g_d
    return ps_g_gs,pd_g_gs,ps_g_gd,pd_g_gd,   gs_g_ps,gs_g_pd,gd_g_ps,gd_g_pd





## Given the condiotnal probabilities like : Pr(pred to b same/same) for each thresholds `t`
def plot_conditionals( t, s_g_s, d_g_s, s_g_d, d_g_d, sty, labels):
    plt.figure(1)
    plt.subplot(221)
    # plt.plot( t, s_g_s, sty, label='Pr(pred to b same/same)'+sty )
    plt.plot( t, s_g_s, sty, label=labels[0] )
    plt.legend( loc='upper right' )
    plt.grid(True)

    plt.subplot(222)
    # plt.plot( t, d_g_s, sty, label='Pr(pred to b diff/same)'+sty )
    plt.plot( t, d_g_s, sty, label=labels[1] )
    plt.legend(loc='upper right' )
    plt.grid(True)

    plt.subplot(223)
    # plt.plot( t, s_g_d, sty, label='Pr(pred to b same/diff)'+sty )
    plt.plot( t, s_g_d, sty, label=labels[2] )
    plt.legend(loc='upper right' )
    plt.grid(True)

    plt.subplot(224)
    # plt.plot( t, d_g_d, sty, label='Pr(pred to b diff/diff)'+sty )
    plt.plot( t, d_g_d, sty, label=labels[3] )
    plt.legend(loc='upper right' )
    plt.grid(True)


def likelihood_ratio(C):
    r_n =  C[0,0] / (C[0,0]+C[0,1]) #prob of being pred as same place given they are same place
    r_d =  C[1,0] / (C[1,0]+C[1,1]) #prob of being pred as same place given that they are different place
    return r_n / r_d



## Given a confusion matrix returns the Mathews correlation score
def mathews_corr_score(C):
    mcc_n = C[0,0]*C[1,1] - C[0,1]*C[1,0]
    mcc_d = (C[0,0]+C[0,1])*(C[0,1]+C[1,1])*(C[1,1]+C[1,0])*(C[1,0]+C[0,0])
    return mcc_n / np.sqrt(mcc_d)





#
# Init Tensorflow
# Init netvlad - def computational graph, load trained model
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')
vgg_obj = VGGDescriptor(b=16)
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_MODEL )



#
# Init DimRed Mapping (Dimensionality Reduction by Learning Invariant Mapping)
dm_vlad_word = tf.placeholder( 'float', [None,None], name='vlad_word' )
net = DimRed.DimRed()
dm_vlad_char = net.fc( dm_vlad_word )
tensorflow_saver2 = tf.train.Saver( net.return_vars() )
tensorflow_saver2.restore( tensorflow_session, PARAM_MODEL_DIM_RED )
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL_DIM_RED, tcolor.ENDC




#
# Renderer (q, [P]. [N])
app = NetVLADRenderer()
# app = TrainRenderer()
print 'Pre-run'
im_batch = None
while im_batch == None: #if queue not sufficiently filled, try again
    im_batch, label_batch = app.step(16)
print 'Queue sufficiently filled'

vlad_collection = []
thumb_collection = []
for n_runs in range(PARAM_N_RUNS):
    C = np.zeros( (len(PARAM_THRESH),2,2) )
    D = np.zeros( (len(PARAM_THRESH),2,2) )
    ps_g_gs = np.zeros( (len(PARAM_THRESH)) )
    pd_g_gs = np.zeros( (len(PARAM_THRESH)) )
    ps_g_gd = np.zeros( (len(PARAM_THRESH)) )
    pd_g_gd = np.zeros( (len(PARAM_THRESH)) )

    gs_g_ps = np.zeros( (len(PARAM_THRESH)) )
    gs_g_pd = np.zeros( (len(PARAM_THRESH)) )
    gd_g_ps = np.zeros( (len(PARAM_THRESH)) )
    gd_g_pd = np.zeros( (len(PARAM_THRESH)) )
    startTime = time.time()
    for n_renders in range(PARAM_N_RENDERS):
        im_batch, label_batch = app.step(16)
        while im_batch == None: #if queue not sufficiently filled, try again
            im_batch, label_batch = app.step(16)

        if n_renders%100 == 0:
            print tcolor.OKBLUE, 'Done run#%3d, render#%4d' %(n_runs, n_renders)
        #Here in im_batch, 0th image is the query, 1-5 are positive samples, 6-16 are negative samples

        im_batch_normalized = normalize_batch( im_batch )
        thumbs = thumbnail_batch( im_batch )
        feed_dict = {tf_x : im_batch_normalized,\
                     is_training:False,\
                     vgg_obj.initial_t: 0}
        tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
        vlad_collection.append(tff_vlad_word)
        thumb_collection.append( thumbs )

        # dim reduction. Also compute the confusion matrix with `tff_vlad_char` as descriptor
        # tff_vlad_char = reduce_dimensions( tff_vlad_word )
        dmm_vlad_char = tensorflow_session.run( dm_vlad_char, feed_dict={dm_vlad_word: tff_vlad_word})



        # Find correct detections and make confusion matrix
        for i in range(len(PARAM_THRESH)):
            C[i,:,:] += get_confusion_mat( tff_vlad_word, PARAM_THRESH[i] )
            D[i,:,:] += get_confusion_mat( dmm_vlad_char, PARAM_THRESH[i] )

    #for each confusion matrix print likelihood ratio
    print ' run#%-4d(%4.2f) : ' %(n_runs, (time.time()-startTime) ),
    for i in range(len(PARAM_THRESH)):
        print '%8.3f(%4.2f)' %(likelihood_ratio(C[i,:,:]), mathews_corr_score(C[i,:,:]) ),
    print ''
    print tcolor.FAIL,'run#%-4d(%4.2f) : ' %(n_runs, (time.time()-startTime) ),
    for i in range(len(PARAM_THRESH)):
        print '%8.3f(%4.2f)' %(likelihood_ratio(D[i,:,:]), mathews_corr_score(D[i,:,:]) ),
    print '', tcolor.ENDC

    # Plot the conditional probabilities which are obtained from the confusion matrix
    for i in range(len(PARAM_THRESH)):
        # s_g_s[i], d_g_s[i], s_g_d[i], d_g_d[i] = conf_mat_analysis( D[i,:,:])
        ps_g_gs[i],pd_g_gs[i],ps_g_gd[i],pd_g_gd[i],   gs_g_ps[i],gs_g_pd[i],gd_g_ps[i],gd_g_pd[i] = conf_mat_analysis( D[i,:,:])

    labels1 = ['Pr(pred to b same/same)', 'Pr(pred to b diff/same)', 'Pr(pred to b same/diff)', 'Pr(pred to b diff/diff)']
    labels2 = ['Pr(same/pred to b same)', 'Pr(same/pred to b diff)', 'Pr(diff/pred to b same)', 'Pr(diff/pred to b diff)' ]
    plot_conditionals( PARAM_THRESH, ps_g_gs,pd_g_gs,ps_g_gd,pd_g_gd, sty='b-', labels=labels1)
    plot_conditionals( PARAM_THRESH, gs_g_ps,gs_g_pd,gd_g_ps,gd_g_pd, sty='r-', labels=labels2)
    plt.show()

    code.interact( local=locals() )





# M = np.vstack( vlad_collection )
# T = np.vstack( thumb_collection )
# npzfname =  'dim_red_training_dat.npz'
# np.savez( npzfname, M=M, thumbs=T )
# print 'Written file : ', npzfname
