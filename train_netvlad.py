""" NetVLAD (cvpr2016) paper. Implementation.
        Basic idea is to learn a 16D representation. Cost function being the
        triplet ranking loss
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
from CartWheelFlow import VGGDescriptor

#
import TerminalColors
tcolor = TerminalColors.bcolors()


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
        tensorboard_prefix = 'tf.logs/netvlad'


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





## Verify the tensorflow computation with numpy computation
def verify_cost( vlad_word, nP, nN, margin ):
    #vlad_word : 16x12288. [q, Px5, Nx10]
    dim = float(vlad_word.shape[1])

    dis_q_P = []
    for i in range(1,1+nP):
        dif = vlad_word[0,:] - vlad_word[i,:]
        dis_q_P.append( np.dot( dif.T, dif ) / dim )


    dis_q_N = []
    for i in range(1+nP,1+nP+nN):
        dif = vlad_word[0,:] - vlad_word[i,:]
        dis_q_N.append( np.dot( dif.T, dif ) / dim )


    cost = max(0, max(dis_q_P) - min(dis_q_N) + margin )
    # code.interact( local=locals() )

    return dis_q_P, dis_q_N, cost

## Set the learning rate for RMSPropOptimizer
def get_learning_rate( n_iteration, base_lr):
    if n_iteration < 60:
        return base_lr
    elif n_iteration >= 100 and n_iteration < 300:
        return base_lr/2.
    elif n_iteration >= 300 and n_iteration < 500:
        return base_lr/4
    elif n_iteration >= 500 and n_iteration < 1000:
        return base_lr/8
    elif n_iteration >= 1000 and n_iteration < 2000:
        return base_lr/16
    else:
        return base_lr/40



#
# Parse Commandline
PARAM_tensorboard_prefix, PARAM_n_write_summary, \
    PARAM_model_save_prefix, PARAM_n_write_tf_model, PARAM_model_restore = parse_cmd_args()
print tcolor.HEADER, 'tensorboard_prefix     : ', PARAM_tensorboard_prefix, tcolor.ENDC
print tcolor.HEADER, 'write_summary every    : ', PARAM_n_write_summary, 'iterations', tcolor.ENDC
print tcolor.HEADER, 'model_save_prefix      : ', PARAM_model_save_prefix, tcolor.ENDC
print tcolor.HEADER, 'write_tf_model every   : ', PARAM_n_write_tf_model, 'iterations', tcolor.ENDC

print tcolor.HEADER, 'model_restore          : ', PARAM_model_restore, tcolor.ENDC




#
# Tensorflow - VGG16-NetVLAD Word
tf_x = tf.placeholder( 'float', [16,240,320,3], name='x' )
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor()
tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)


#
# Tensorflow - Cost function (Triplet Loss)
# fitting_loss = 0

nP = 5
nN = 10
margin = 0.1
fitting_loss = vgg_obj.svm_hinge_loss( tf_vlad_word, nP=nP, nN=nN, margin=margin )
regularization_loss = tf.add_n( slim.losses.get_regularization_losses() )
tf_cost = regularization_loss + fitting_loss



#
# Gradient Computation
# make a grad computation op and an assign_add op
tf_lr = tf.placeholder( 'float', shape=[], name='learning_rate' )
tensorflow_opt = tf.train.RMSPropOptimizer( tf_lr )


trainable_vars = tf.trainable_variables()
for vv in trainable_vars:
    print 'name=', vv.name, 'shape=' ,vv.get_shape().as_list()
print '# of trainable_vars : ', len(trainable_vars)


# borrowed from https://github.com/tensorflow/tensorflow/issues/3994 (issue 3994 of tensorflow)
accum_vars = [tf.Variable( tf.zeros_like(tv.initialized_value()), name=tv.name.replace( '/', '__' ).replace(':', '__'), trainable=False ) for tv in trainable_vars ] #create new set of variables to accumulate gradients
zero_op = [ tv.assign( tf.zeros_like(tv) ) for tv in accum_vars  ] #set above var zeros
grad_var = tensorflow_opt.compute_gradients( tf_cost, trainable_vars )
accum_op = [ accum_vars[i].assign_add(gv[0]) for i,gv in enumerate(grad_var)  ]
train_step = tensorflow_opt.apply_gradients( [(accum_vars[i], gv[1])  for i,gv in enumerate(grad_var)] )

# Cummulate total cost
cummulative_cost = tf.Variable( 0, dtype=tf.float32, trainable=False )
zero_cc_cost_op = cummulative_cost.assign( tf.zeros_like(cummulative_cost) )
accum_cc_cost_op = cummulative_cost.assign_add( tf_cost )




# Summary
tf.summary.scalar( 'cc_cost', cummulative_cost )
tf.summary.scalar( 'lr', tf_lr )
# list trainable variables
for vv in trainable_vars:
    tf.summary.histogram( vv.name, vv )
    tf.summary.scalar( 'sparsity_var'+vv.name, tf.nn.zero_fraction(vv) )
for gg in accum_vars:
    tf.summary.histogram( gg.name, gg )
    tf.summary.scalar( 'sparsity_grad'+gg.name, tf.nn.zero_fraction(gg) )

# tf.summary.histogram( 'xxxx_inputs', tf_x )




#
# Init Tensorflow - Xavier initializer, session
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
# Tensorboard and Saver
# Tensorboard
summary_writer = tf.summary.FileWriter( PARAM_tensorboard_prefix, tensorflow_session.graph )
summary_op = tf.summary.merge_all()


# Saver
with tf.device( '/cpu:0' ):
    tensorflow_saver = tf.train.Saver()


#
# Setup NetVLAD Renderer - This renderer is custom made for NetVLAD training
app = NetVLADRenderer()


#
# Iterations
while True:

    startTime = time.time()


    # for i in range(16):
    #     print '(%6.1f,%6.1f,%6.1f)    ' %(label_batch[i,0],label_batch[i,1],label_batch[i,2]),
    #
    #     fname = 'dump/'+str(l)+'_'+str(i)+'.jpg'
    #     print 'Write : ', fname
    #     cv2.imwrite( fname, im_batch[i,:,:,:].astype('uint8'))
    # print "\n"
    # time.sleep(5)

    tensorflow_session.run([zero_op,zero_cc_cost_op]) #set gradient_cummulator and cost_cummulator to zero

    mini_batch = 24
    mbatch_total_cost = 0
    # accumulate gradient
    for _ in range(mini_batch):
        im_batch, label_batch = app.step(16)
        while im_batch == None: #if queue not sufficiently filled, try again
            im_batch, label_batch = app.step(16)


        feed_dict = {tf_x : im_batch,\
                     is_training:True
                    }
        # tff_cost, tff_word, _grad_ = tensorflow_session.run( [tf_cost, tf_vlad_word, accum_op], feed_dict=feed_dict)
        # _dis_q_P, _dis_q_N, _cost = verify_cost( tff_word, nP, nN, margin )
        # print tff_cost, _cost
        tff_cost, _grad_, tff_cc_cost, regloss = tensorflow_session.run( [tf_cost, accum_op, accum_cc_cost_op, regularization_loss], feed_dict=feed_dict)
        mbatch_total_cost = mbatch_total_cost + tff_cost



    _, summary_exec = tensorflow_session.run( [train_step,summary_op], feed_dict={tf_lr: get_learning_rate(tf_iteration, 0.001) } )

    print '%3d(%8.2fms) : cost=%8.3f cc_cost=%8.3f fit_loss=%8.3f reg_loss=%8.3f' %(tf_iteration, 1000.*(time.time() - startTime), mbatch_total_cost, tff_cc_cost, (tff_cc_cost-regloss*mini_batch), regloss*mini_batch)


    # Periodically Save Models
    if tf_iteration % PARAM_n_write_tf_model == 0 and tf_iteration > 0:
        print 'Snapshot model : ', PARAM_model_save_prefix, tf_iteration
        tensorflow_saver.save( tensorflow_session, PARAM_model_save_prefix, global_step=tf_iteration )

    if tf_iteration % PARAM_n_write_summary == 0:
        print 'Write Summary : ', PARAM_tensorboard_prefix
        summary_writer.add_summary( summary_exec, tf_iteration )






    tf_iteration = tf_iteration + 1
    # code.interact( local=locals() )
