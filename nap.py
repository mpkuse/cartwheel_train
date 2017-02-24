""" Neural Appearence based Place model (NAP)
        Using the learned netvlad to find nearest neighbours from
        past frames in current sequence in the high dimensioanl space (8192-dim).

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 17th Feb, 2017
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import code
import argparse
import glob
from annoy import AnnoyIndex

import tensorflow as tf
import tensorflow.contrib.slim as slim
from CartWheelFlow import VGGDescriptor

#
import TerminalColors
tcolor = TerminalColors.bcolors()

PARAM_MODEL = 'tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'

PARAM_BAG_DUMP = 'other_seqs/Lip6IndoorDataSet_npz/ttt_'#'./bag_dump/bag11/dji_sdk_'

PARAM_BAG_START = 1
PARAM_BAG_END   = len( glob.glob(PARAM_BAG_DUMP+'*.npz'))-2
PARAM_BAG_STEP = 1#20


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


# Init netvlad - def computational graph, load trained model
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')
vgg_obj = VGGDescriptor(b=1)
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_MODEL )

# Collection bins
word_stack = []
image_stack = []

# Load all images, compute descriptors for each and store in array
prev_word = None
prev_word_indx = -1
plt.ion()
fig=plt.figure()
plt.axis('tight')

for ind in range( PARAM_BAG_START, PARAM_BAG_END, PARAM_BAG_STEP ):
    npzFileName = PARAM_BAG_DUMP+str(ind)+'.npz'
    data = np.load( npzFileName )
    A = data['A'] #cv2.flip( data['A'], 0 )

    print tcolor.OKGREEN, 'Load NPZFile : ', npzFileName, 'of ', PARAM_BAG_END, tcolor.ENDC




    im_batch = np.zeros( (1,A.shape[0],A.shape[1],A.shape[2]) )
    im_batch[0,:,:,:] = A.astype('float32')
    im_batch_normalized = normalize_batch( im_batch )

    startTime = time.time()
    feed_dict = {tf_x : im_batch_normalized,\
                 is_training:False,\
                 vgg_obj.initial_t: 0}
    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
    print '%6.2fms' %((time.time() - startTime)*1000.), tcolor.OKGREEN, 'Load NPZFile : ', npzFileName, 'of ', PARAM_BAG_END, tcolor.ENDC

    word_stack.append( tff_vlad_word )
    image_stack.append( A.astype('uint8') )

    if prev_word is not None:
        diff_word = (tff_vlad_word - prev_word)
        fig.clear()
        plt.xlim( [-0.05,0.05])
        plt.ylim( [0,5000])
        plt.hist( diff_word[0,:] )
        # plt.draw()
        # plt.show(block=False)
        plt.pause(0.001)
        print 'curr=%d, keyframe=%d :' %(ind, prev_word_indx),\
                    np.dot(tff_vlad_word[0,:] , prev_word[0,:]), \
                    np.mean(abs(diff_word))*1000., \
                    np.median(abs(diff_word))*1000., \
                    np.std(abs(diff_word))*1000.


    # if ind % 35 == 1 or np.max(prev_word-tff_vlad_word)>0.01:
    if prev_word == None or np.dot(tff_vlad_word[0,:], prev_word[0,:]) < 0.8:
        print 'Key frame word replaced : frame=', ind
        prev_word = tff_vlad_word
        prev_word_indx = ind
        prev_word_im = im_batch[0,:,:,:].astype('uint8')
        print 'Writing keyframe : ', 'dump/%d.jpg' %(prev_word_indx)
        cv2.imwrite( 'dump/%d.jpg' %(prev_word_indx), cv2.cvtColor(prev_word_im,cv2.COLOR_BGR2RGB ) )

    cv2.imshow( 'win', cv2.cvtColor( A.astype('uint8'), cv2.COLOR_RGB2BGR ) )
    cv2.waitKey(0)




# Make KD-tree
#   TODO: look at techniqures to build KD-tree index on the fly for ANN search
startTime = time.time()
words_db  = np.vstack( word_stack ) #Nx8192
t_ann = AnnoyIndex( words_db.shape[1], metric='angular'  )
for i in range( words_db.shape[0] ):
    t_ann.add_item( i, words_db[i,:] )
t_ann.build(10)
print '%6.2fms' %((time.time() - startTime)*1000.), ' Built ANN Index with %d items' %(t_ann.get_n_items())


# Find NN for each image in netvlad descriptor space.
C = np.zeros( (t_ann.get_n_items(), t_ann.get_n_items()) )
nn_window = 15
for i,word in enumerate(word_stack):
    nn_indx, nn_dist = t_ann.get_nns_by_vector(word[0,:], nn_window, include_distances=True )
    print 'nn_dist : ', np.round(nn_dist,2)
    print 'nn_indx : ', nn_indx
    # cont = raw_input( "Press Enter to Continue" )
    C[i,nn_indx] = np.exp( np.negative(nn_dist) )
    # code.interact(local=locals())
cv2.imshow( 'C', np.round(C*255).astype('uint8') )
cv2.waitKey(0)
