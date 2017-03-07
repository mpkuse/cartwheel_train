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
from annoy import AnnoyIndex


from PandaRender import TrainRenderer
from CartWheelFlow import VGGDescriptor
import DimRed

#
import TerminalColors
tcolor = TerminalColors.bcolors()


#
# Params
PARAM_MODEL = 'tf.logs/netvlad_angular_loss_w_mini_dev/model-4000'
sl = PARAM_MODEL.rfind( '/' )
PARAM_DB_PREFIX = PARAM_MODEL[:sl] + '/viz_22/'

PARAM_ENABLE_DIM_RED = True
PARAM_MODEL_DIM_RED = 'tf.logs/siamese_dimred_fc/model-400'

N_RENDERS = -1#4000 #set this to -1 to disable synthetic images

PARAM_BAG_DUMP = [\
                #   './bag_dump/bag3/dji_sdk_', \
                #   './bag_dump/bag8/dji_sdk_', \
                #   './bag_dump/bag9/dji_sdk_', \
                #   './bag_dump/bag10/dji_sdk_',\
                #   './bag_dump/bag11/dji_sdk_',\
                #   './bag_dump/bag21/dji_sdk_',\
                  './bag_dump/bag22/dji_sdk_'
                # './other_seqs/Lip6OutdoorDataSet_npz/outdoor_kennedylong'
                # 'other_seqs/Lip6IndoorDataSet_npz/ttt_'
                    # 'other_seqs/data_collection_20100901_npz/c0/img_'
                  ]

PARAM_BAG_START = [1 for bag in PARAM_BAG_DUMP] # [1,1,1,1,1,1]
PARAM_BAG_END  = [ len( glob.glob(bag+'*.npz'))-2 for bag in PARAM_BAG_DUMP ] #500
# PARAM_BAG_END  = [ 10000 for bag in PARAM_BAG_DUMP ] #500
PARAM_BAG_STEP = [2 for bag in PARAM_BAG_DUMP]#[20,20,20,20,20,20]

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
        # print 'Factors : ', all_factors
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


# words_db is Nx8192 matrix
def make_appearance_confusion_matrix( words_db, nn_window ):
    startTime = time.time()
    t_ann = AnnoyIndex( words_db.shape[1], metric='angular'  )
    for i in range( words_db.shape[0] ):
        t_ann.add_item( i, words_db[i,:] )
    t_ann.build(10)
    print '%6.2fms' %((time.time() - startTime)*1000.), ' Built ANN Index with %d items each %d-dim' %(t_ann.get_n_items(), words_db.shape[1])

    # Find NN for each image in netvlad descriptor space.
    C = np.zeros( (t_ann.get_n_items(), t_ann.get_n_items()) )

    for i in range(words_db.shape[0]):
        nn_indx, nn_dist = t_ann.get_nns_by_vector(words_db[i,:], nn_window, include_distances=True )
        # print 'nn_dist : ', np.round(nn_dist,2)
        # print 'nn_indx : ', nn_indx
        # cont = raw_input( "Press Enter to Continue" )
        C[i,nn_indx] = np.exp( np.negative(nn_dist) )
    # cv2.imshow( 'C', np.round(C*255).astype('uint8') )
    ret_C = np.zeros( (1,C.shape[0],C.shape[1],1))
    ret_C[0,:,:,0] = C
    return C, ret_C

## TODO: similar to above, but with using only previously scene examples and with neighbours nearer than threshold
def make_appearance_confusion_matrix_nhood_size( words_db, nn_dist_thresh ):
    """ """
    p=0

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
# Init DimRed Mapping (Dimensionality Reduction by Learning Invariant Mapping)
if PARAM_ENABLE_DIM_RED:
    dm_vlad_word = tf.placeholder( 'float', [None,None], name='vlad_word' )
    net = DimRed.DimRed()
    dm_vlad_char = net.fc( dm_vlad_word )
    tensorflow_saver2 = tf.train.Saver( net.return_vars() )
    tensorflow_saver2.restore( tensorflow_session, PARAM_MODEL_DIM_RED )
    print tcolor.OKGREEN,'Restore model from : ', PARAM_MODEL_DIM_RED, tcolor.ENDC



#
# Collection bins
word_stack = []
char_stack = []
image_stack = []
thumbnail_stack = []




#
# Images from Renderer
if N_RENDERS > 0 :
    app = TrainRenderer(queue_warning=False)
    INS = len(thumbnail_stack)
    print 'Generating Samples from Renderer'
    for itr in range(N_RENDERS):

        im_batch = None
        while im_batch==None:
            im_batch, label_batch = app.step(1)

        im_batch_normalized = normalize_batch( im_batch )

        startTime = time.time()
        feed_dict = {tf_x : im_batch_normalized,\
                     is_training:False,\
                     vgg_obj.initial_t: 0}

        tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
        word_stack.append( tff_vlad_word )

        if PARAM_ENABLE_DIM_RED:
            # Dimensionality Reduction
            dmm_vlad_char = tensorflow_session.run( dm_vlad_char, feed_dict={dm_vlad_word: tff_vlad_word})
            char_stack.append(dmm_vlad_char)


        # db_image = cv2.cvtColor( (255.0*im_batch_normalized[0,:,:,:]).astype('uint8'), cv2.COLOR_RGB2BGR  )
        db_image = cv2.cvtColor( im_batch[0,:,:,:].astype('uint8'), cv2.COLOR_RGB2BGR  )
        image_stack.append( db_image )
        thumbnail_stack.append( cv2.resize( db_image, (0,0), fx=0.2, fy=0.2 ) )

        if itr%500 == 0:
            print tcolor.OKGREEN,'Done Iteration %d in %5.2fms' %(itr, (time.time()-startTime)*1000.), tcolor.ENDC
    print tcolor.HEADER, 'RENDER', INS, len(thumbnail_stack)-1, tcolor.ENDC
else:
    print tcolor.WARNING, 'No synthetically rendered images', tcolor.ENDC

#
# Real Images
print 'Samples from Real Images'
for u in range(len(PARAM_BAG_DUMP)):
    INS = len(thumbnail_stack)
    for ind in range(PARAM_BAG_START[u], PARAM_BAG_END[u], PARAM_BAG_STEP[u]):
        npzFileName = PARAM_BAG_DUMP[u]+str(ind)+'.npz'
        # print tcolor.OKGREEN, 'Load NPZFile : ', npzFileName, 'of ', PARAM_BAG_END[u], tcolor.ENDC
        data = np.load( npzFileName )
        A = data['A'] #cv2.flip( data['A'], 0 )
        im_batch = np.zeros( (1,A.shape[0],A.shape[1],A.shape[2]) )
        im_batch[0,:,:,:] = A.astype('float32')

        im_batch_normalized = normalize_batch( im_batch )

        feed_dict = {tf_x : im_batch_normalized,\
                     is_training:False,\
                     vgg_obj.initial_t: 0}

        tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict )
        word_stack.append( tff_vlad_word )

        if PARAM_ENABLE_DIM_RED:
            # Dimensionality Reduction
            dmm_vlad_char = tensorflow_session.run( dm_vlad_char, feed_dict={dm_vlad_word: tff_vlad_word})
            char_stack.append(dmm_vlad_char)

        # db_image = cv2.cvtColor( (255.0*im_batch_normalized[0,:,:,:]).astype('uint8'), cv2.COLOR_RGB2BGR  )
        db_image = cv2.cvtColor( im_batch[0,:,:,:].astype('uint8'), cv2.COLOR_RGB2BGR  )
        image_stack.append(db_image )
        thumbnail_stack.append( cv2.resize( db_image, (0,0), fx=0.2, fy=0.2 ) )

    print tcolor.HEADER, PARAM_BAG_DUMP[u], INS, len(thumbnail_stack)-1, tcolor.ENDC


with tf.variable_scope( 'netVLAD', reuse=True ):
    vlad_c = tf.get_variable( 'vlad_c' )#, [K,D], initializer=tf.contrib.layers.xavier_initializer()) #KxD

# Collection bins
print tcolor.WARNING, '#items in stacks [word_stack,char_stack,image_stack,thumbnail_stack] : [%d,%d,%d,%d]' %(len(word_stack),len(char_stack),len(image_stack),len(thumbnail_stack) )



#
# Write Sprite Image from thumbnail_stack
summary_writer = tf.summary.FileWriter( PARAM_DB_PREFIX )
sprite_image = makeSprite(thumbnail_stack)
im_h, im_w, _ = thumbnail_stack[0].shape
cv2.imwrite( os.path.join( PARAM_DB_PREFIX, 'SPRITE.png'), sprite_image )
print 'Written Sprite Image : ', os.path.join( PARAM_DB_PREFIX, 'SPRITE.png' )




#
# Process collection bins and produce output for embedding-visualization
word_stack_embeding = np.vstack( word_stack )
tf_embedding = tf.Variable( word_stack_embeding, name='netvlad_descriptors' )

if PARAM_ENABLE_DIM_RED:
    char_stack_embeding = np.vstack( char_stack )
    tf_embedding_char = tf.Variable( char_stack_embeding, name='netvlad_char_descriptors' )
    saver_arg = [tf_embedding, tf_embedding_char, vlad_c]
else:
    saver_arg = [tf_embedding, vlad_c]

tensorflow_session.run( tf.global_variables_initializer() )

embeding_saver = tf.train.Saver( saver_arg )

config = projector.ProjectorConfig()

# Set sprite image for full vlad word 8192-dim
embedding = config.embeddings.add()
embedding.tensor_name=tf_embedding.name
embedding.sprite.image_path =  os.path.join( PARAM_DB_PREFIX, 'SPRITE.png')
embedding.sprite.single_image_dim.extend([im_w,im_h]) #note that it asks col-count and then row-count

if PARAM_ENABLE_DIM_RED:
    # Set sprite for 128-dim reduced vlad word, viz vlad_char
    embedding = config.embeddings.add()
    embedding.tensor_name=tf_embedding_char.name
    embedding.sprite.image_path =  os.path.join( PARAM_DB_PREFIX, 'SPRITE.png')
    embedding.sprite.single_image_dim.extend([im_w,im_h]) #note that it asks col-count and then row-count


projector.visualize_embeddings( summary_writer, config)
embeding_saver.save( tensorflow_session, os.path.join( PARAM_DB_PREFIX, "embeding.ckpt" ) )

os.makedirs( PARAM_DB_PREFIX+'/im' ) #TODO: if  /im already exist do not attempt to recreate it. Give a warning
for i,im in enumerate(image_stack):
    cv2.imwrite( PARAM_DB_PREFIX+'/im/'+str(i)+'.png', im )

# Write embedding as npy array for easy access
print 'Write file : ', PARAM_DB_PREFIX+'netvlad_descriptors.npy'
np.save(PARAM_DB_PREFIX+'netvlad_descriptors.npy', word_stack_embeding )

if PARAM_ENABLE_DIM_RED:
    print 'Write file : ', PARAM_DB_PREFIX+'netvlad_char_descriptors.npy'
    np.save(PARAM_DB_PREFIX+'netvlad_char_descriptors.npy', char_stack_embeding )


# Neural Apearence Based NN - do this only when 1 bag
if N_RENDERS < 0 and len(PARAM_BAG_DUMP)==1:
    word_stack_embeding  = np.vstack( word_stack )
    startTime = time.time()
    print tcolor.OKGREEN, 'Appearence Confusion Mat for vlad_word', tcolor.ENDC
    nn_window = 75
    C, ret_C = make_appearance_confusion_matrix(word_stack_embeding, nn_window )
    print C.shape, ' confusion matrix created in %4.2fms for %d-dimensions and %d-nn' %( (time.time() - startTime)*1000., word_stack_embeding.shape[1], nn_window )
    c_fname = PARAM_DB_PREFIX+'/make_appearance_confusion_matrix.png'
    cv2.imwrite( c_fname, (C*255).astype('uint8' ) ) #TODO: write a color-mapped file
    print 'Written Image : ', c_fname

    # Appearence confusion map using the vlad_char
    if PARAM_ENABLE_DIM_RED:
        char_stack_embeding = np.vstack( char_stack )
        startTime = time.time()
        print tcolor.OKGREEN, 'Appearence Confusion Mat for vlad_char', tcolor.ENDC
        nn_window = 75
        C, ret_C = make_appearance_confusion_matrix(char_stack_embeding, nn_window )
        print C.shape, ' confusion matrix created in %4.2fms for %d-dimensions and %d-nn' %( (time.time() - startTime)*1000., char_stack_embeding.shape[1], nn_window )
        c_fname = PARAM_DB_PREFIX+'/make_appearance_confusion_matrix_dim_red.png'
        cv2.imwrite( c_fname, (C*255).astype('uint8' ) ) #TODO: write a color-mapped file
        print 'Written Image : ', c_fname




print 'tensorboard --logdir %s' %(PARAM_DB_PREFIX)
