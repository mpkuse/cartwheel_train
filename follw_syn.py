""" Render 1 view at a time and do prediction.
    Prediction works with hard-assgnment into 1 of 120 classes.
"""

import argparse
import time
import code

from PandaRender import TestRenderer
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import CartWheelFlow as puf
import SpaGrid

#
import TerminalColors
tcolor = TerminalColors.bcolors()


# Given a square matrix, substract mean and divide std dev
def zNormalized( M ):
    M_mean = np.mean(M) # scalar
    M_std = np.std(M)
    if M_std < 0.0001 :
        return M

    M_statSquash = (M - M_mean)/(M_std+.0001)
    return M_statSquash


# Parse CMD-arg
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--model_restore", help="Path of model file for restore. This file path is \
                                split(-) and last number is set as iteration count. \
                                Absense of this will lead to xavier init")
args = parser.parse_args()

if args.model_restore:
    PARAM_model_restore = args.model_restore
else:
    PARAM_model_restore = None


# Set up tensorflow prediction
puf_obj = puf.CartWheelFlow(trainable_on_device='/cpu:0')

tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
tf_infer_op = puf_obj.resnet50_inference(tf_x, is_training=False)  # Define these inference ops on all the GPUs
print tcolor.OKGREEN, 'Setting up TF.Graph ... Done..!', tcolor.ENDC


# TF Session
tensorflow_session = tf.Session( config=tf.ConfigProto(allow_soft_placement=True) )


# Load trained model
print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
tensorflow_saver = tf.train.Saver()
tensorflow_saver.restore( tensorflow_session, PARAM_model_restore )

a__ = PARAM_model_restore.find('-')
n__ = int(PARAM_model_restore[a__+1:])
print tcolor.OKGREEN,'Restore Iteration : ', n__, tcolor.ENDC

tf.train.start_queue_runners(sess=tensorflow_session)



# Setup renderer
app = TestRenderer()

l = 0
im_batch = np.zeros((1,240,320,3))
sg = SpaGrid.SpaGrid()
while True:
    im, y = app.step()

    if im is not None:
        # cv2.imwrite( 'dump/'+str(l)+'.jpg', im )
        s=0
        # zNormalized im
        im_batch[0,:,:,0] = zNormalized( im[:,:,0] )
        im_batch[0,:,:,1] = zNormalized( im[:,:,1] )
        im_batch[0,:,:,2] = zNormalized( im[:,:,2] )



        # do tf prediction
        aa_out = tensorflow_session.run( [tf_infer_op], feed_dict={ tf_x:im_batch } ) #1x4x12x1
        aa_out = aa_out[0]



        # compare with GT
        x_ = y[0]
        y_ = y[1]
        z_ = y[2]
        class_n = sg.cord2Indx( x_, y_)



        # print aa_out.argmax(), CLASS
        sortedIndx = aa_out[0].argsort()[-7:]
        print aa_out[0].argsort()[-7:], '::', class_n

        # cv2.imshow( 'win', im.astype('uint8') )
        # cv2.waitKey(0)


        qq = aa_out[0]
        qq_softmax = np.exp( qq - np.max(qq) ) / sum( np.exp( qq - np.max(qq) ) )
        plt.plot( qq_softmax )
        plt.show()

        #code.interact(local=locals())



    l = l + 1
