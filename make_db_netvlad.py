"""
    Create a DB with a few (~1000s) images with their GT positions and fingerprint
    To run this script you need a trained model and the network. Essentially this
    script with draw N samples compute its fingerprint and save it along with images and GT

    Author  : Manohar Kuse <mpkuse@ust.hk>
    Created : 24th Jan, 2017
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

from PandaRender import TrainRenderer
from CartWheelFlow import VGGDescriptor

from annoy import AnnoyIndex

#
import TerminalColors
tcolor = TerminalColors.bcolors()



#TODO: Write a function to parse arguments
PARAM_MODEL = 'tf.logs/netvlad_inp_normed_angular_loss/model-4000'
sl = PARAM_MODEL.rfind( '/' )
PARAM_DB_PREFIX = PARAM_MODEL[:sl] + '/db2/'
PARAM_BATCHSIZE = 16 #usually less than 16
PARAM_N_RENDERS = 300

print tcolor.HEADER, 'PARAM_MODEL     : ', PARAM_MODEL, tcolor.ENDC
print tcolor.HEADER, 'PARAM_DB_PREFIX : ', PARAM_DB_PREFIX, tcolor.ENDC
print tcolor.HEADER, 'PARAM_BATCHSIZE : ', PARAM_BATCHSIZE, tcolor.ENDC
print tcolor.HEADER, 'PARAM_N_RENDERS : ', PARAM_N_RENDERS, tcolor.ENDC



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


def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        im_batch_normalized[b,:,:,:] = rgbnormalize( im_batch[b,:,:,:] )
    return im_batch_normalized


#
# Init Renderer - Using this renderer as we want a few random poses to make a DB
app = TrainRenderer()


#
# Define tensorflow computation-graph
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor()
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)





#
# Load stored Weights
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_MODEL )


#
# Prep Directory to store
if not os.path.exists( PARAM_DB_PREFIX ):
    print tcolor.OKBLUE, 'Create Directory : ', PARAM_DB_PREFIX, tcolor.ENDC
    os.makedirs( PARAM_DB_PREFIX )
    os.makedirs( PARAM_DB_PREFIX+'/im' )
else:
    print tcolor.WARNING, 'Directory exists : ', PARAM_DB_PREFIX, '....It might contains old files', tcolor.ENDC



word_stack = []
label_stack = []
thumbnail_stack = []
im_indx = 0
for itr in range(PARAM_N_RENDERS):
    startTime = time.time()

    batch_size = PARAM_BATCHSIZE
    im_batch = None
    label_batch = None
    while im_batch==None:
        im_batch, label_batch = app.step(batch_size)

    im_batch_normalized = normalize_batch( im_batch )

    feed_dict = {tf_x : im_batch,\
                 is_training:False,\
                 vgg_obj.initial_t: 0}

    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict=feed_dict )
    print 'tff_vlad_word.shape : ', tff_vlad_word.shape



    # Write a) Im, b) vlad vec, c) labels
    for j in range(batch_size):
        fname = PARAM_DB_PREFIX+'im/'+str(im_indx)+'.jpg'
        print 'Write : ', fname
        db_image = cv2.cvtColor( im_batch[j,:,:,:].astype('uint8'), cv2.COLOR_RGB2BGR  )
        thumbnail_stack.append( cv2.resize( db_image, (0,0), fx=0.2, fy=0.2 ) )
        cv2.imwrite( fname, db_image )
        im_indx = im_indx + 1
    word_stack.append( tff_vlad_word )
    label_stack.append( label_batch )



    print 'iteration %d in %4.2f ms' %( itr, (time.time()-startTime)*1000. )



print '---------------------------'
word_stack_2  = np.vstack( word_stack )
label_stack_2 = np.vstack( label_stack )

print 'Images Path : ', PARAM_DB_PREFIX+'/im/'
print tcolor.OKGREEN, 'Total Images Written : ', str(PARAM_BATCHSIZE*PARAM_N_RENDERS), tcolor.ENDC


with open( PARAM_DB_PREFIX+'/vlad_word.pickle', 'w') as handle:
    print 'Writing : ',word_stack_2.shape, ' : ', PARAM_DB_PREFIX+'/vlad_word.pickle'
    pickle.dump( word_stack_2, handle, protocol=pickle.HIGHEST_PROTOCOL )
    print tcolor.OKGREEN, 'Done..', tcolor.ENDC

with open( PARAM_DB_PREFIX+'/label.pickle', 'w') as handle:
    print 'Writing : ', label_stack_2.shape, ' : ', PARAM_DB_PREFIX+'/label.pickle'
    pickle.dump( label_stack_2, handle, protocol=pickle.HIGHEST_PROTOCOL )
    print tcolor.OKGREEN, 'Done..', tcolor.ENDC



#
# Build KD-tree : Approximate NN
print  'Building KD Tree for Approximate Nearest Neighbour Search'
ann_f = word_stack_2.shape[1]
ann_t = AnnoyIndex( ann_f, metric='euclidean' )
for l in range( word_stack_2.shape[0] ):
    ann_t.add_item(l, word_stack_2[l,:] )


ann_t.build(10)
ann_t.save( PARAM_DB_PREFIX+'/KDTree.ann' )
print tcolor.OKGREEN, 'Done writing ann index', tcolor.ENDC
