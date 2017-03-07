""" Neural Appearence based Place model (NAP)
        Using the learned netvlad to find nearest neighbours from
        past frames in current sequence in the high dimensioanl space (8192-dim).

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 17th Feb, 2017
        Major R : 1st  Mar, 2017
"""

import cv2
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, dok_matrix
import scipy.sparse as spp
import matplotlib.pyplot as plt
import time
import code
import argparse
import glob
# from annoy import AnnoyIndex
# import kdtree
import VPTree

import tensorflow as tf
import tensorflow.contrib.slim as slim
from CartWheelFlow import VGGDescriptor
import DimRed


#
import TerminalColors
tcolor = TerminalColors.bcolors()

PARAM_MODEL = 'tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
PARAM_MODEL_DIM_RED = 'tf.logs/siamese_dimred_fc/model-400'

# PARAM_BAG_DUMP = 'other_seqs/Lip6IndoorDataSet_npz/ttt_'#'./bag_dump/bag11/dji_sdk_'
# PARAM_BAG_DUMP = 'other_seqs/Lip6OutdoorDataSet_npz/outdoor_kennedylong'
PARAM_BAG_DUMP = 'other_seqs/kitti_dataset_npz/00/'


PARAM_BAG_START = 0
PARAM_BAG_END   = len( glob.glob(PARAM_BAG_DUMP+'*.npz'))-2
PARAM_BAG_STEP = 1#20

PARAM_DEBUG = True


def print_sparse_matrix( M, msg='M' ):
    ind = M.nonzero()
    for i in range(len(ind[0])):
        print '%s[%3d,%3d] = %6.4f' %(msg, ind[0][i], ind[1][i], M[ind[0][i], ind[1][i]] )


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

#
# Init netvlad - def computational graph, load trained model
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')
vgg_obj = VGGDescriptor(b=1)
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



# Load all images, compute descriptors for each and store in KD-tree
# tree = kdtree.create( dimensions=128 )
treeroot = None
if PARAM_DEBUG:
    d_CHAR_ary = [] #ONLY FOR DEBUGGING
    time_tree_insertion = []
    time_tree_nn_search = []
    APPEARENCE_CONFUSION = np.zeros( (PARAM_BAG_END+1, PARAM_BAG_END+1) ) #ONLY FOR DEBUGGING
    for jk in range(PARAM_BAG_START):
        d_CHAR_ary.append( np.zeros(128) )

rb_PRIOR = dok_matrix( (1000000,1) )
rb_PRIOR[0] = 1.0
rb_OBSER = lil_matrix( (1000000,1) )
# rb_POSTERIOR = lil_matrix( (1000000,1) )
for ind in range( PARAM_BAG_START, PARAM_BAG_END, PARAM_BAG_STEP ):
    # Read Image
    npzFileName = PARAM_BAG_DUMP+str(ind)+'.npz'
    data = np.load( npzFileName )
    A = data['A'] #cv2.flip( data['A'], 0 )
    # print tcolor.OKGREEN, 'Load NPZFile : ', npzFileName, 'of ', PARAM_BAG_END, tcolor.ENDC
    print tcolor.OKGREEN, 'Load NPZFile : ', npzFileName, 'of ', PARAM_BAG_END, tcolor.ENDC


    # Position Index
    pos_index = ind #TODO Ideally, should index by approx distance. Probly think of getting this with IMU prop

    ############# descriptors compute starts
    # VLAD Computation
    d_compute_time_ms = []
    startTime = time.time()
    im_batch = np.zeros( (1,A.shape[0],A.shape[1],A.shape[2]) )
    im_batch[0,:,:,:] = A.astype('float32')
    im_batch_normalized = normalize_batch( im_batch )
    d_compute_time_ms.append( (time.time() - startTime)*1000. )

    startTime = time.time()
    feed_dict = {tf_x : im_batch_normalized,\
                 is_training:False,\
                 vgg_obj.initial_t: 0}
    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
    d_WORD = tff_vlad_word[0,:]
    d_compute_time_ms.append( (time.time() - startTime)*1000. )


    # Dim Reduction
    startTime = time.time()
    dmm_vlad_char = tensorflow_session.run( dm_vlad_char, feed_dict={dm_vlad_word: tff_vlad_word})
    dmm_vlad_char = dmm_vlad_char
    d_CHAR = dmm_vlad_char[0,:]
    d_compute_time_ms.append( (time.time() - startTime)*1000. )

    ###### END of descriptor compute : d_WORD, d_CHAR, d_compute_time_ms[] ###############
    print '[%6.2fms] Descriptor Computation' %(sum(d_compute_time_ms))
    # d_compute_time_ms = [ time_for_color_normalize, time_for_vlad_comp, time_for_dim_red ]
    # if PARAM_DEBUG:
        # d_CHAR_ary.append(d_CHAR)


    # KDTREE / VPTree - Insert
    # tree.add( d_CHAR[0:3] )
    startAddTime = time.time()
    node = VPTree.NDPoint( d_CHAR, pos_index )
    if treeroot is None:
        treeroot = VPTree.InnerProductTree(node, 0.27) # #PARAM this is mu
    else:
        treeroot.add_item( node )
    print '[%6.2fms] Added to InnerProductTree' %( (time.time() - startTime)*1000. )
    if PARAM_DEBUG:
            time_tree_insertion.append(  (time.time() - startTime)*1000. )



    # Tree Nearest Neighbour Query
    startAddTime = time.time()
    q = VPTree.NDPoint( d_CHAR, -1 )
    # all_nn= VPTree.get_nearest_neighbors( treeroot, q, k=10 )
    all_nn= VPTree.get_all_in_range( treeroot, q, tau=0.23 ) # #PARAM this is tau
    print tcolor.OKBLUE+'[%6.2fms] NN Query' %(  (time.time() - startTime)*1000. ), tcolor.ENDC,
    if PARAM_DEBUG:
            time_tree_nn_search.append(  (time.time() - startTime)*1000. )

    print pos_index, ":",
    for nn in all_nn:
        print nn[1].idx, '(%5.3f), ' %(nn[0]),
        # if PARAM_DEBUG:
            # APPEARENCE_CONFUSION[ pos_index ,nn[1].idx] = 1.0 #np.exp( np.negative(nn[0]) )
    print ''


    # Filter Nearest Neighbours - Recursive Bayesian Filter
    startRBF = time.time()
    print_sparse_matrix( rb_PRIOR, 'PRIOR' )

    # Step-1 : Record Observation pdf
    rb_OBSER = dok_matrix( (1000000,1) )
    for nn in all_nn:
        # print 'rb_OBSER[%d] = %6.2f' %(nn[1].idx, np.exp( np.negative(nn[0]) ) )
        rb_OBSER[nn[1].idx] = np.exp( np.negative(nn[0]) )
    # rb_OBSER = rb_OBSER.tocoo() / rb_OBSER.tocoo().sum()
    rb_OBSER = rb_OBSER / rb_OBSER.sum()
    print_sparse_matrix( rb_OBSER, 'LKEHD' )

    # Step-2 : Posterior = Likelihood x Prior
    # rb_POSTERIOR = rb_OBSER.multiply( rb_PRIOR )
    rb_POSTERIOR = rb_OBSER + rb_PRIOR
    rb_POSTERIOR = rb_POSTERIOR / rb_POSTERIOR.sum()
    print_sparse_matrix( rb_POSTERIOR, 'POSTR' )

    # Step-3 : Propagate Posterior and set is as PRIOR (for next step)
    tmp = rb_POSTERIOR.todok()
    rb_PRIOR = dok_matrix( (1000000,1) )
    for key in tmp.keys():
        # print 'rb_PRIOR[ %d,%d ] = rb_POSTERIOR[ %d, %d ]' %(key[0]+1, key[1],  key[0] , key[1])
        rb_PRIOR[ key[0]+1, key[1] ] = rb_POSTERIOR[ key[0] , key[1] ]
        rb_PRIOR[ key[0]+1+1, key[1] ] = 0.47*rb_POSTERIOR[ key[0] , key[1] ]
        rb_PRIOR[ key[0], key[1] ] = 0.47*rb_POSTERIOR[ key[0] , key[1] ]
    rb_PRIOR = rb_PRIOR / rb_PRIOR.sum()

    print tcolor.OKBLUE+'[%6.2fms] Recursive Bayesian Filter' %( (time.time() - startRBF)*1000. ), tcolor.ENDC

    code.interact( local=locals() )





    cv2.imshow( 'win', cv2.cvtColor( A.astype('uint8'), cv2.COLOR_RGB2BGR ) )
    key = cv2.waitKey(10)
    if key == 27:
        break
    # code.interact( local=locals() )


if PARAM_DEBUG:
    cv2.imwrite( 'APPEARENCE_CONFUSION.png', (APPEARENCE_CONFUSION*255).astype('uint8' ) ) #TODO: write a color-mapped file
    VPTree.visualize( treeroot, max_lvl=20 )
    plt.plot( time_tree_insertion )
    plt.plot( time_tree_nn_search )
    plt.show(  )
print 'Done...!'
