"""
    Create a few images from renderer, few from each of the bags.
    Compute the netvlad vector for all the images and store it for visualization.
    Things to store: a) netvlad vectors, b) Original image and spirit

    Note: The labels does not matter here for visualization, so we wont store them
    we are just trying to see the how close or far images are compared to others.

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 12th Feb, 2017
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import code
import argparse
import os
import pickle
import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.tensorboard.plugins import projector #for t-SNE visualization

from PandaRender import TrainRenderer
from CartWheelFlow import VGGDescriptor
#
import TerminalColors
tcolor = TerminalColors.bcolors()


#
# Params
PARAM_MODEL = 'tf.logs/netvlad_angular_loss/model-7800'
sl = PARAM_MODEL.rfind( '/' )
PARAM_DB_PREFIX = PARAM_MODEL[:sl] + '/viz1/'

N_RENDERS = 2000

PARAM_BAG_DUMP = ['./bag_dump/bag3/dji_sdk_', './bag_dump/bag8/dji_sdk_', './bag_dump/bag9/dji_sdk_', './bag_dump/bag10/dji_sdk_', './bag_dump/bag11/dji_sdk_' ]
PARAM_BAG_START = [1,1,1,1,1]
PARAM_BAG_END  = [ len( glob.glob(bag+'*.npz'))-2 for bag in PARAM_BAG_DUMP ] #500
PARAM_BAG_STEP = [20,20,20,20,20]

#TODO: Add assert here to ensure BAG params are of equal length

def factors(n):
    return [i for i in range(1, int(np.sqrt(n+100))) if not n%i]

def _debug( msg ):
    _q=0
    # print 'DEBUG :', msg

def makeSprite( thumbnail_stack ):
    print tcolor.OKGREEN, 'Creating Sprite Image', tcolor.ENDC
    while True:
        n = len(thumbnail_stack)
        r, c, ch = thumbnail_stack[0].shape
        _debug('---Try---')
        print 'Total %d images in stack, each of dim (%d,%d)' %(n,r,c)

        #make size of sprite is 8192x8192
        # print 'Total Images that can fit in the spirit : ', np.round( (8192*8192)/(r*c) )
        #Estimate the sprite image dimension before proceeding, if in appropriate
        #padd images
        all_factors = factors(n)
        number_of_row = factors(n)[-1]
        number_of_col = n / number_of_row
        print 'Factors : ', all_factors
        _debug( '%d thumbnails in a row and %d thumbnails in col' %(number_of_row, number_of_col) )
        _debug( 'estimated spirite image dim : %dx%d' %(r*number_of_row, c*number_of_col) )
        if r*number_of_row < 8192 and c*number_of_col < 8192: #8192x8192 is the largest supported stride image
            _debug('---Done--')
            break
        _debug('Pad 1 image and estimate again...')
        thumbnail_stack.append( np.zeros( (r,c,ch)) )


    v = []
    # print  'number of images per row : ', nrow
    thumbnail_sp = np.array_split( thumbnail_stack, number_of_row )
    for sp in thumbnail_sp:
        sprite_row = np.concatenate(sp, axis=1 )
        # print sprite_row.shape
        v.append( sprite_row )

    sprite = np.concatenate(v, axis=0 )
    print tcolor.OKBLUE, 'Sprite Image Dim : ', sprite.shape, tcolor.ENDC
    return sprite


#
# Setup Tensorflow with trained weights
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')
vgg_obj = VGGDescriptor(b=1)
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL, tcolor.ENDC
tensorflow_saver.restore( tensorflow_session, PARAM_MODEL )


#
# Collection bins
word_stack = []
image_stack = []
thumbnail_stack = []




#
# Images from Renderer
app = TrainRenderer(queue_warning=False)
INS = len(thumbnail_stack)
print 'Generating Samples from Renderer'
for itr in range(N_RENDERS):

    im_batch = None
    while im_batch==None:
        im_batch, label_batch = app.step(1)


    startTime = time.time()
    feed_dict = {tf_x : im_batch,\
                 is_training:False,\
                 vgg_obj.initial_t: 0}

    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
    word_stack.append( tff_vlad_word )
    db_image = cv2.cvtColor( im_batch[0,:,:,:].astype('uint8'), cv2.COLOR_RGB2BGR  )
    image_stack.append( db_image )
    thumbnail_stack.append( cv2.resize( db_image, (0,0), fx=0.2, fy=0.2 ) )

    if itr%500 == 0:
        print tcolor.OKGREEN,'Done Iteration %d in %5.2fms' %(itr, (time.time()-startTime)*1000.), tcolor.ENDC
print tcolor.HEADER, 'RENDER', INS, len(thumbnail_stack)-1, tcolor.ENDC


#
# Real Images
print 'Samples from Real Images'
for u in range(len(PARAM_BAG_DUMP)):
    INS = len(thumbnail_stack)
    for ind in range(PARAM_BAG_START[u], PARAM_BAG_END[u], PARAM_BAG_STEP[u]):
        npzFileName = PARAM_BAG_DUMP[u]+str(ind)+'.npz'
        # print tcolor.OKGREEN, 'Load NPZFile : ', npzFileName, 'of ', PARAM_BAG_END[u], tcolor.ENDC
        data = np.load( npzFileName )
        A = cv2.flip( data['A'], 0 )
        im_batch = np.zeros( (1,A.shape[0],A.shape[1],A.shape[2]) )
        im_batch[0,:,:,:] = A.astype('float32')

        feed_dict = {tf_x : im_batch,\
                     is_training:False,\
                     vgg_obj.initial_t: 0}

        tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
        word_stack.append( tff_vlad_word )
        db_image = cv2.cvtColor( im_batch[0,:,:,:].astype('uint8'), cv2.COLOR_RGB2BGR  )
        image_stack.append(db_image )
        thumbnail_stack.append( cv2.resize( db_image, (0,0), fx=0.2, fy=0.2 ) )

    print tcolor.HEADER, PARAM_BAG_DUMP[u], INS, len(thumbnail_stack)-1, tcolor.ENDC




#
# Process collection bins and produce output for embedding-visualization
word_stack_embeding  = np.vstack( word_stack )
sprite_image = makeSprite(thumbnail_stack)
tf_embedding = tf.Variable( word_stack_embeding, name='netvlad_descriptors' )
tensorflow_session.run( tf.global_variables_initializer() )

summary_writer = tf.summary.FileWriter( PARAM_DB_PREFIX )
embeding_saver = tf.train.Saver( [tf_embedding] )
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name=tf_embedding.name
cv2.imwrite( os.path.join( PARAM_DB_PREFIX, 'SPRITE.png'), sprite_image )
print 'Writen Sprite Image : ', os.path.join( PARAM_DB_PREFIX, 'SPRITE.png' )
embedding.sprite.image_path =  os.path.join( PARAM_DB_PREFIX, 'SPRITE.png')
im_h, im_w, _ = thumbnail_stack[0].shape
embedding.sprite.single_image_dim.extend([im_w,im_h]) #note that it asks col-count and then row-count
projector.visualize_embeddings( summary_writer, config)
embeding_saver.save( tensorflow_session, os.path.join( PARAM_DB_PREFIX, "embeding.ckpt" ) )

os.makedirs( PARAM_DB_PREFIX+'/im' )
for i,im in enumerate(image_stack):
    cv2.imwrite( PARAM_DB_PREFIX+'/im/'+str(i)+'.png', im )
