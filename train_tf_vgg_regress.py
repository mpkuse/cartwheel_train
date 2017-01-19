""" Training with TF-SLIM. As a regression model """

import argparse
import time

# Usual Math and Image Processing
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

#
import TerminalColors
tcolor = TerminalColors.bcolors()

from PandaRender import TrainRenderer
import CartWheelFlow

def parse_cmd_args():
    """Parse Arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tensorboard_prefix", help="Path for tensorboard")
    parser.add_argument("-s", "--model_save_prefix", help="Path for saving model. If not specified will be same as tensorboard_prefix")
    parser.add_argument("-r", "--model_restore", help="Path of model file for restore. This file path is \
                                    split(-) and last number is set as iteration count. \
                                    Absense of this will lead to xavier init")

    parser.add_argument("-wsu", "--write_summary", help="Write summary after every N iteration (default:20)")
    parser.add_argument("-wmo", "--write_tf_model", help="Write tf model after every N iteration (default:5000)")
    args = parser.parse_args()


    # Prefix path to for `SummaryWriter`
    if args.tensorboard_prefix:
    	tensorboard_prefix = args.tensorboard_prefix
    else:
        tensorboard_prefix = 'tf.logs/default'


    if args.write_summary:
        write_summary = int(args.write_summary) #TODO: check this is not negative or zero
    else:
        write_summary = 20

    # Prefix path for `Saver`
    if args.model_save_prefix:
    	model_save_prefix = args.model_save_prefix
    else:
        model_save_prefix = tensorboard_prefix+'/model'

    if args.write_tf_model:
        write_tf_model = int(args.write_tf_model) #TODO: check this is not negative or zero
    else:
        write_tf_model = 5000


    if args.model_restore:
        model_restore = args.model_restore
    else:
        model_restore = None




    return tensorboard_prefix, write_summary, model_save_prefix, write_tf_model, model_restore


def get_learning_rate( n_iteration, base_lr):
    if n_iteration < 300:
        return base_lr
    elif n_iteration >= 500 and n_iteration < 800:
        return base_lr/2.
    elif n_iteration >= 800 and n_iteration < 1500:
        return base_lr/4
    elif n_iteration >= 1500 and n_iteration < 2500:
        return base_lr/8
    elif n_iteration >= 2500 and n_iteration < 4000:
        return base_lr/16
    else:
        return base_lr/40

def _print_trainable_variables():
    """ Print all trainable variables. List is obtained using tf.trainable_variables() """
    var_list = tf.trainable_variables()
    print '--Trainable Variables--', 'length = ', len(var_list)
    total_n_nums = []
    for vr in var_list:
        shape = vr.get_shape().as_list()
        n_nums = np.prod(shape)
        total_n_nums.append( n_nums )
        print tcolor.OKGREEN, vr.name, shape, n_nums, tcolor.ENDC

    print tcolor.OKGREEN, 'Total Trainable Params (floats): ', sum( total_n_nums )
    print 'Not counting the pop_mean and pop_varn as these were set to be non trainable', tcolor.ENDC
    print '--Trainable Variables--', 'length = ', len(var_list)


def define_loss( pred_x, pred_y, pred_z, pred_yaw, label_x, label_y, label_z, label_yaw ):
    """ defines the l2-loss """
    loss_x = tf.reduce_mean( tf.square( tf.sub( pred_x, label_x ) ), name='loss_x' )
    loss_y = tf.reduce_mean( tf.square( tf.sub( pred_y, label_y ) ), name='loss_y' )
    loss_z = tf.reduce_mean( tf.square( tf.sub( pred_z, label_z ) ), name='loss_z' )
    loss_yaw = tf.reduce_mean( tf.square( tf.sub( pred_yaw, label_yaw ) ), name='loss_yaw' )

    xpy = tf.add( tf.mul( loss_x, 1.0 ), tf.mul( loss_y, 1.0 ), name='x__p__y')
    zpyaw = tf.add(tf.mul( loss_z, 1.0 ), tf.mul( loss_yaw, 0.5 ), name='z__p__yaw' )
    fitting_loss = tf.sqrt( tf.add(xpy,zpyaw,name='full_sum'), name='aggregated_loss'  )

    return fitting_loss



#
# Parse Commandline Arguments
PARAM_tensorboard_prefix, PARAM_n_write_summary, \
    PARAM_model_save_prefix, PARAM_n_write_tf_model, PARAM_model_restore = parse_cmd_args()
print tcolor.HEADER, 'tensorboard_prefix     : ', PARAM_tensorboard_prefix, tcolor.ENDC
print tcolor.HEADER, 'write_summary every    : ', PARAM_n_write_summary, 'iterations', tcolor.ENDC
print tcolor.HEADER, 'model_save_prefix      : ', PARAM_model_save_prefix, tcolor.ENDC
print tcolor.HEADER, 'write_tf_model every   : ', PARAM_n_write_tf_model, 'iterations', tcolor.ENDC

print tcolor.HEADER, 'model_restore          : ', PARAM_model_restore, tcolor.ENDC



#
# Make VGG Predictor
tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
tf_label_x = tf.placeholder( 'float',   [None,1], name='label_x')
tf_label_y = tf.placeholder( 'float',   [None,1], name='label_y')
tf_label_z = tf.placeholder( 'float',   [None,1], name='label_z')
tf_label_yaw = tf.placeholder( 'float',   [None,1], name='label_yaw')
is_training = tf.placeholder( tf.bool, [], name='is_training')
tf_learning_rate = tf.placeholder( 'float', shape=[], name='learning_rate' )


vgg_pred_x, vgg_pred_y, vgg_pred_z, vgg_pred_yaw = CartWheelFlow.VGGFlow().vgg16( tf_x, is_training )

_print_trainable_variables()



# loss = tf.nn.l2_loss( tf.sub( tf.mul( 500.0, vgg_pred ), tf_label ) )
loss = define_loss( vgg_pred_x, vgg_pred_y, vgg_pred_z, vgg_pred_yaw, tf_label_x, tf_label_y, tf_label_z, tf_label_yaw)
regularization_loss = tf.add_n( slim.losses.get_regularization_losses() )
total_loss = regularization_loss + loss

#
# Optimizer
# TODO: Can use 3-steps to get hold of gradients. viz. opt.compute_gradients(); opt.apply_gradients()
train_op = tf.train.AdamOptimizer( tf_learning_rate ).minimize( total_loss )



#
# Session
tensorflow_session = tf.Session()


# If PARAM_model_restore is none means init from scratch.
if PARAM_model_restore == None:
    print tcolor.OKGREEN,'global_variables_initializer() : xavier', tcolor.ENDC
    tensorflow_session.run( tf.global_variables_initializer() )
    tf_iteration = 0
else:
    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver.restore( tensorflow_session, PARAM_model_restore )

    a__ = PARAM_model_restore.find('-')
    n__ = int(PARAM_model_restore[a__+1:])
    print tcolor.OKGREEN,'Restore Iteration : ', n__, tcolor.ENDC
    tf_iteration =  n__#the number after `-`. eg. model-40000, 40000 will be set as iteration



#
# Init Renderer
app = TrainRenderer()



tf.summary.scalar( 'loss', total_loss )
tf.summary.scalar( 'lr', tf_learning_rate )

# Tensorboard
summary_writer = tf.summary.FileWriter( PARAM_tensorboard_prefix, tensorflow_session.graph )
summary_op = tf.summary.merge_all()


# Saver
tensorflow_saver = tf.train.Saver()



#
# Iterations
tf_iteration = 0
while True:
    startTime = time.time()


    batchsize = 30
    im_batch, label_batch = app.step(batchsize=batchsize)

    #TODO: zNormalize im_batch
    lr = get_learning_rate( tf_iteration, 0.0005 )
    feed_dict = {tf_x:im_batch,\
                tf_label_x:label_batch[:,0:1],\
                tf_label_y:label_batch[:,1:2],\
                tf_label_z:label_batch[:,2:3],\
                tf_label_yaw:label_batch[:,3:4],\
                is_training:True,\
                tf_learning_rate:lr}

    _, a_, b_, c_, summary_exec = tensorflow_session.run( [train_op, total_loss, regularization_loss, loss, summary_op], feed_dict=feed_dict )

    elapsedTimeMS = (time.time() - startTime)*1000.
    print '%3d(%4dms) : loss=%2.4f, lr=%2.4f, reg_loss=%2.4f, fit_loss=%2.4f' %(tf_iteration, int(elapsedTimeMS), a_, lr, b_, c_ )


    # Periodically Save Models
    if tf_iteration % PARAM_n_write_tf_model == 0 and tf_iteration > 0:
        print 'Snapshot model : ', PARAM_model_save_prefix, tf_iteration
        tensorflow_saver.save( tensorflow_session, PARAM_model_save_prefix, global_step=tf_iteration )

    if tf_iteration % PARAM_n_write_summary == 0 and tf_iteration > 0:
        print 'Write Summary : ', PARAM_tensorboard_prefix
        summary_writer.add_summary( summary_exec, tf_iteration )


    app.taskMgr.step()
    tf_iteration = tf_iteration + 1
