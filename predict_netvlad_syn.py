""" Given an image, computes its fingerprint. Does the ANN query and finds the most similar image.
    This script loads the 3d model rendering an image. Thus, this script is a testing script on
    synthetically generated images.

    Created : 25th Jan, 2017
    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
"""


import cv2
import numpy as np
import time
import argparse
import os
import pickle
import code
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim

from PandaRender import TrainRenderer
from CartWheelFlow import VGGDescriptor

from annoy import AnnoyIndex

#
import TerminalColors
tcolor = TerminalColors.bcolors()



PARAM_MODEL = 'tf.logs/netvlad_logsumexp_loss/model-17500'
PARAM_DB_PREFIX = 'tf.logs/netvlad_logsumexp_loss/db2/'


#
# Init Tensorflow prediction
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
# Load ANN Index
with open( PARAM_DB_PREFIX+'vlad_word.pickle', 'r' ) as handle:
    print 'Read : ', PARAM_DB_PREFIX+'vlad_word.pickle'
    words_db = pickle.load( handle )

t_ann = AnnoyIndex( words_db.shape[1], metric='euclidean'  )

for i in range( words_db.shape[0] ):
    t_ann.add_item( i, words_db[i,:] )

print 'Rebuild ANN Index' #TODO: Figure out why t_ann.load() does not work
t_ann.build(10)



#
# Init Renderer
app = TrainRenderer(queue_warning=False)
while True:
    im, label = app.step(1)
    im_float = im[0,:,:,:]
    im_uint8 = cv2.cvtColor( im_float.astype('uint8'), cv2.COLOR_RGB2BGR  )


    feed_dict = {tf_x : im,\
                 is_training:False
                }


    startTime = time.time()
    tff_vlad_word = tensorflow_session.run( tf_vlad_word, feed_dict=feed_dict )
    nn_indx, nn_dist = t_ann.get_nns_by_vector( tff_vlad_word[0,:], 5, include_distances=True )
    print 'predict_time = %5.2f ms' %( (time.time() - startTime)*1000. )
    print 'nn_dist : ', nn_dist
    print 'nn_indx : ', nn_indx
    for en,h in enumerate(nn_indx):
        im_name = PARAM_DB_PREFIX+'im/'+str(h)+'.jpg'
        # print 'Read Image : ', im_name
        q_im = cv2.imread( im_name )
        cv2.imshow( str(en), q_im )
    cv2.imshow( 'win', im_uint8 )


    # plt.bar( range(0,tff_vlad_word.shape[1]), tff_vlad_word[0,:] )
    # plt.draw()
    # plt.show( )

    key = cv2.waitKey(0)

    if key == 27: #ESC
        break

print 'Quit...!'
