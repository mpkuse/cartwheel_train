""" NetVLAD (cvpr2016) paper. Implementation.
        Basic idea is to learn a 16D representation. Cost function being the
        triplet ranking loss

        Added code for pairwise loss



        Author  : Manohar Kuse <mpkuse@ust.hk>
        Created : 12th Jan, 2017
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import code
import argparse
import os
import datetime

from collections import OrderedDict
import json

import tensorflow as tf
import tensorflow.contrib.slim as slim

TF_MAJOR_VERSION = int(tf.__version__.split('.')[0])
TF_MINOR_VERSION = int(tf.__version__.split('.')[1])

# from PandaRender import NetVLADRenderer
from CartWheelFlow import VGGDescriptor
from TimeMachineRender import TimeMachineRender
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer

#
import TerminalColors
tcolor = TerminalColors.bcolors()

# import pyqtgraph as pg

def parse_cmd_args():
    """Parse Arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tensorboard_prefix", help="Path for tensorboard")
    parser.add_argument("-s", "--model_save_prefix", help="Path for saving model. If not specified will be same as tensorboard_prefix")
    parser.add_argument("-f", "--config_file", help="File name of config-file (file should be a .json)" )
    parser.add_argument("-r", "--model_restore", help="Path of model file for restore. This file path is \
                                    split(-) and last number is set as iteration count. \
                                    Absense of this will lead to xavier init")
    parser.add_argument("-rn", "--restore_iteration_number", help="Overide the iteration number from -r argument. \
                                    If not specified, will read iteration num from model file name. Will be \
                                    active only when restoring")

    parser.add_argument("-wsu", "--write_summary", help="Write summary after every N iteration (default:5)")
    parser.add_argument("-wmo", "--write_tf_model", help="Write tf model after every N iteration (default:250)")

    parser.add_argument( '--imshow', action='store_true' )
    args = parser.parse_args()


    # Prefix path to for `SummaryWriter`
    if args.tensorboard_prefix:
    	tensorboard_prefix = args.tensorboard_prefix
    else:
        tensorboard_prefix = 'tf.logs/netvlad'


    if args.write_summary:
        write_summary = int(args.write_summary) #TODO: check this is not negative or zero
    else:
        write_summary = 5

    # Prefix path for `Saver`
    if args.model_save_prefix:
    	model_save_prefix = args.model_save_prefix
    else:
        model_save_prefix = tensorboard_prefix+'/model'

    if args.write_tf_model:
        write_tf_model = int(args.write_tf_model) #TODO: check this is not negative or zero
    else:
        write_tf_model = 250


    if args.model_restore:
        model_restore = args.model_restore
    else:
        model_restore = None


    if args.restore_iteration_number:
        restore_iteration_number = int(args.restore_iteration_number)
    else:
        restore_iteration_number = -1

    if args.config_file:
        # Check Existence of the file
        if os.path.exists( args.config_file ):
            config_filename = args.config_file
        else:
            print tcolor.FAIL, 'config_file does not exist. Quitting', tcolor.ENDC
    else:
        print tcolor.FAIL, 'config_file is required to be mentioned. Quitting', tcolor.ENDC
        quit()

    if args.imshow:
        imshow_bool = True
    else:
        imshow_bool = False

    return tensorboard_prefix, write_summary, model_save_prefix, write_tf_model, model_restore, restore_iteration_number, config_filename, imshow_bool





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

## M is a 2D matrix
def zNormalize(M):
    return (M-M.mean())/(M.std()+0.0001)

def rgbnormalize( im ):
    im_R = im[:,:,0].astype('float32')
    im_G = im[:,:,1].astype('float32')
    im_B = im[:,:,2].astype('float32')
    # S = im_R + im_G + im_B
    S = abs(im_R) + abs(im_G) + abs(im_B)
    out_im = np.zeros(im.shape)
    out_im[:,:,0] = im_R / (S+.0001)
    out_im[:,:,1] = im_G / (S+.0001)
    out_im[:,:,2] = im_B / (S+.0001)

    return out_im


def normalize_batch( im_batch ):
    im_batch_normalized = np.zeros(im_batch.shape)
    for b in range(im_batch.shape[0]):
        for ch in range(im_batch.shape[3]):
                im_batch_normalized[b,:,:,ch] = zNormalize( im_batch[b,:,:,ch])
        # im_batch_normalized[b,:,:,0] = zNormalize( im_batch[b,:,:,0])
        # im_batch_normalized[b,:,:,1] = zNormalize( im_batch[b,:,:,1])
        # im_batch_normalized[b,:,:,2] = zNormalize( im_batch[b,:,:,2])
        # im_batch_normalized[b,:,:,:] = rgbnormalize( im_batch_normalized[b,:,:,:] )

    # cv2.imshow( 'org_', (im_batch[0,:,:,:]).astype('uint8') )
    # cv2.imshow( 'out_', (im_batch_normalized[0,:,:,:]*255.).astype('uint8') )
    # cv2.waitKey(0)
    # code.interact(local=locals())
    return im_batch_normalized



#
# Parse Commandline
PARAM_tensorboard_prefix, PARAM_n_write_summary, \
    PARAM_model_save_prefix, PARAM_n_write_tf_model, \
    PARAM_model_restore, PARAM_restore_iteration_number,\
    PARAM_config_json_filename, PARAM_imshow = parse_cmd_args()
print tcolor.HEADER, 'tensorboard_prefix     : ', PARAM_tensorboard_prefix, tcolor.ENDC
print tcolor.HEADER, 'write_summary every    : ', PARAM_n_write_summary, 'iterations', tcolor.ENDC
print tcolor.HEADER, 'model_save_prefix      : ', PARAM_model_save_prefix, tcolor.ENDC
print tcolor.HEADER, 'write_tf_model every   : ', PARAM_n_write_tf_model, 'iterations', tcolor.ENDC

print tcolor.HEADER, 'model_restore          : ', PARAM_model_restore, tcolor.ENDC
print tcolor.HEADER, 'restore_iteration_n    : ', PARAM_restore_iteration_number, tcolor.ENDC
print tcolor.HEADER, 'config_file            : ', PARAM_config_json_filename, tcolor.ENDC


#
# Runtime Parameters

# Load JSON
print 'Open JSON-config file: ', PARAM_config_json_filename
with open( PARAM_config_json_filename ) as json_data:
    FILE_PARAMS = json.load( json_data )

nP =                 FILE_PARAMS['nP']
nN =                 FILE_PARAMS['nN']
margin =             FILE_PARAMS['MARGIN']
scale_gamma =        FILE_PARAMS['SCALE_GAMMA']
MINI_BATCH_SIZE =    FILE_PARAMS['MINI_BATCH_SIZE']
NET_TYPE =           FILE_PARAMS['NET_TYPE'] #currently ["vgg6", "resnet6"]
FITTING_LOSS_TYPE =  FILE_PARAMS['FITTING_LOSS_TYPE'] # currently ["soft_angular_ploss", "weakly_supervised_ranking_loss" ]
ENABLE_POS_SET_DEV = FILE_PARAMS['ENABLE_POS_SET_DEV']
PARAM_K =            FILE_PARAMS['PARAM_K']
ENABLE_IMSHOW = PARAM_imshow
#TODO: Validate these values. Currently working on trust that all these are OK values.
# Dont break my trust... :).

# Default code- Can Overide if need be
# nP = 8
# nN = 8
# margin = 0.1
# scale_gamma = 0.07
# MINI_BATCH_SIZE = 24
# NET_TYPE = "resnet6" #currently ["vgg6", "resnet6"]
# FITTING_LOSS_TYPE = "soft_angular_ploss" # currently ["soft_angular_ploss", "weakly_supervised_ranking_loss" ]
# ENABLE_POS_SET_DEV = True



#
# Tensorflow - NetVLAD Word
learning_batch_size = 1+nP+nN #Note: nP and nN is not well tested with pandarenderer. However it is ok with timemachine renderer
tf_x = tf.placeholder( 'float', [learning_batch_size,240,320,3], name='x' ) #this has to be 3 if training with color images
is_training = tf.placeholder( tf.bool, [], name='is_training')


vgg_obj = VGGDescriptor(K=PARAM_K, D=256, N=60*80, b=learning_batch_size)
# tf_vlad_word = vgg_obj.vgg16(tf_x, is_training)
tf_vlad_word = vgg_obj.network(tf_x, is_training, net_type=NET_TYPE )


#
# Tensorflow - Cost function

#--- a) Fitting Cost
# fitting_loss = vgg_obj.svm_hinge_loss( tf_vlad_word, nP=nP, nN=nN, margin=margin )
# fitting_loss = vgg_obj.soft_ploss( tf_vlad_word, nP=nP, nN=nN, margin=margin ) #keep margin as 10

if FITTING_LOSS_TYPE == "soft_angular_ploss":
    fitting_loss = vgg_obj.soft_angular_ploss( tf_vlad_word, nP=nP, nN=nN, margin=margin ) #margin as 0.2

if FITTING_LOSS_TYPE == "weakly_supervised_ranking_loss":
    fitting_loss = vgg_obj.weakly_supervised_ranking_loss( tf_vlad_word, nP=nP, nN=nN, margin=margin ) #margin as 0.2


#--- b) Positive Set deviation
# TODO Instead of setting this to zero, do not add it to tf_cost. This ways, it will eval to a value which can be visualized and compared.
pos_set_dev = vgg_obj.positive_set_std_dev( tf_vlad_word, nP=nP, nN=nN, scale_gamma=scale_gamma )
# if ENABLE_POS_SET_DEV:
#     pos_set_dev = vgg_obj.positive_set_std_dev( tf_vlad_word, nP=nP, nN=nN, scale_gamma=scale_gamma )
# else:
#     pos_set_dev = tf.constant( 0.0 )

#--- c) regularization. regularization gamma is set in tf.slim ie, in class VGGDescriptor
if TF_MAJOR_VERSION == 0:
    regularization_loss = tf.add_n( slim.losses.get_regularization_losses() )
else:
    regularization_loss = tf.add_n(  tf.losses.get_regularization_losses()  )

if ENABLE_POS_SET_DEV:
    tf_cost = regularization_loss + fitting_loss + pos_set_dev
else:
    tf_cost = regularization_loss + fitting_loss

for vv in tf.trainable_variables():
    print 'name=', vv.name, 'shape=' ,vv.get_shape().as_list()
print '# of trainable_vars : ', len(tf.trainable_variables())
# quit()


#
# Gradient Computation
# make a grad computation op and an assign_add op
tf_lr = tf.placeholder( 'float', shape=[], name='learning_rate' )
tensorflow_opt = tf.train.RMSPropOptimizer( tf_lr )
# tensorflow_opt = tf.train.AdamOptimizer( tf_lr )


trainable_vars = tf.trainable_variables()



# borrowed from https://github.com/tensorflow/tensorflow/issues/3994 (issue 3994 of tensorflow)
# ops to accumulate gradients
accum_vars = [tf.Variable( tf.zeros_like(tv.initialized_value()), name=tv.name.replace( '/', '__' ).replace(':', '__'), trainable=False ) for tv in trainable_vars ] #create new set of variables to accumulate gradients
zero_op = [ tv.assign( tf.zeros_like(tv) ) for tv in accum_vars  ] #set above var zeros
grad_var = tensorflow_opt.compute_gradients( tf_cost, trainable_vars )
accum_op = [ accum_vars[i].assign_add(gv[0]) for i,gv in enumerate(grad_var)  ]
train_step = tensorflow_opt.apply_gradients( [(accum_vars[i], gv[1])  for i,gv in enumerate(grad_var)] )

# Cummulate total cost
cumel_tf_cost = tf.Variable( 0, dtype=tf.float32, trainable=False )
zero_tf_cost = cumel_tf_cost.assign( tf.zeros_like(cumel_tf_cost) )
accum_tf_cost = cumel_tf_cost.assign_add( tf_cost )

# Cummulate regularization loss
cumel_reg_loss  = tf.Variable( 0, dtype=tf.float32, trainable=False )
zero_reg_loss  = cumel_reg_loss.assign( tf.zeros_like(cumel_reg_loss ) )
accum_reg_loss  = cumel_reg_loss.assign_add( regularization_loss )

# Cummulate fitting loss
cumel_fit_loss = tf.Variable( 0, dtype=tf.float32, trainable=False )
zero_fit_loss  = cumel_fit_loss.assign( tf.zeros_like(cumel_fit_loss) )
accum_fit_loss  = cumel_fit_loss.assign_add( fitting_loss )

# Cummulate positive set deviation
cumel_pos_set_dev = tf.Variable( 0, dtype=tf.float32, trainable=False )
zero_pos_set_dev  = cumel_pos_set_dev.assign( tf.zeros_like(cumel_pos_set_dev) )
accum_pos_set_dev = cumel_pos_set_dev.assign_add( pos_set_dev )


# Summary
tf.summary.scalar( 'cumel_tf_cost', cumel_tf_cost )
tf.summary.scalar( 'cumel_reg_loss', cumel_reg_loss )
tf.summary.scalar( 'cumel_fit_loss', cumel_fit_loss )
tf.summary.scalar( 'cumel_pos_set_dev', cumel_pos_set_dev )
tf.summary.scalar( 'lr', tf_lr )
# list trainable variables
for vv in trainable_vars:
    tf.summary.histogram( vv.name, vv )
    tf.summary.scalar( 'sparsity_var'+vv.name, tf.nn.zero_fraction(vv) )
for gg in accum_vars:
    tf.summary.histogram( gg.name, gg )
    tf.summary.scalar( 'sparsity_grad'+gg.name, tf.nn.zero_fraction(gg) )

# tf.summary.histogram( 'xxxx_inputs', tf_x )

tf_batch_success_ratio = tf.placeholder( 'float', shape=[], name='batch_success_ratio' )
tf.summary.scalar( 'batch_success_ratio', tf_batch_success_ratio )

if TF_MAJOR_VERSION >= 1:
    summary_text = tf.summary.text( 'tag1', tf.convert_to_tensor('Hello World msg') )

#
# Init Tensorflow - Xavier initializer, session
# tensorflow_session = tf.Session()
tensorflow_session = tf.Session( config=tf.ConfigProto(log_device_placement=False, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) )

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=tensorflow_session, coord=coord)

#
# Tensorboard and Saver
# Tensorboard
summary_writer = tf.summary.FileWriter( PARAM_tensorboard_prefix, tensorflow_session.graph )
summary_op = tf.summary.merge_all()


# Saver
tensorflow_saver = tf.train.Saver()



# If PARAM_model_restore is none means init from scratch.
if PARAM_model_restore == None:
    print tcolor.OKGREEN,'global_variables_initializer() : xavier', tcolor.ENDC
    tensorflow_session.run( tf.global_variables_initializer() )
    tf_iteration = 0
else:
    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver.restore( tensorflow_session, PARAM_model_restore )

    if PARAM_restore_iteration_number <= -1:
        a__ = PARAM_model_restore.find('-')
        n__ = int(PARAM_model_restore[a__+1:])
        print tcolor.OKGREEN,'Restore Iteration : ', n__, tcolor.ENDC
        tf_iteration =  n__#the number after `-`. eg. model-40000, 40000 will be set as iteration
    else:
        tf_iteration = PARAM_restore_iteration_number



#
# Plotter - pyqtgraph
# plt.ion()
plt_pos_writer_file = PARAM_tensorboard_prefix+'/pos_'+str(max(0,tf_iteration) )
plt_neg_writer_file = PARAM_tensorboard_prefix+'/neg_'+str(max(0,tf_iteration) )
print tcolor.HEADER, 'Open file ', plt_pos_writer_file, ' to write positive losses for each mini-batch for every iteration', tcolor.ENDC
print tcolor.HEADER, 'Open file ', plt_neg_writer_file, ' to write negative losses for each mini-batch for every iteration', tcolor.ENDC
plt_pos_writer = open( plt_pos_writer_file , 'w+', 0 )
plt_neg_writer = open( plt_neg_writer_file , 'w+', 0 )


# Write config file for later debugging (with timestamp)
FILE_PARAMS = OrderedDict()
FILE_PARAMS['nP'] = nP
FILE_PARAMS['nN'] = nN
FILE_PARAMS['MARGIN'] = margin
FILE_PARAMS['SCALE_GAMMA'] = scale_gamma
FILE_PARAMS['MINI_BATCH_SIZE'] = MINI_BATCH_SIZE
FILE_PARAMS['NET_TYPE'] = NET_TYPE
FILE_PARAMS['FITTING_LOSS_TYPE'] = FITTING_LOSS_TYPE
FILE_PARAMS['ENABLE_POS_SET_DEV'] = ENABLE_POS_SET_DEV
FILE_PARAMS['PARAM_K'] = PARAM_K

# Print
for __knk in FILE_PARAMS.keys():
    print tcolor.BOLD, __knk, tcolor.ENDC, ' : ', FILE_PARAMS[__knk]

out_debug_config_filename = PARAM_tensorboard_prefix+'/config_%s.json' %( str(datetime.datetime.now()) )
print tcolor.HEADER, 'Open file ', out_debug_config_filename, tcolor.ENDC
with open( out_debug_config_filename , 'w' ) as fp:
    json.dump( FILE_PARAMS, fp, indent=4 )

# End writing config for debugging



#
# Setup NetVLAD Renderer - This renderer is custom made for NetVLAD training.
# It renderers 16 images at a time. 1st im is query image. Next nP images are positive samples. Next nN samples are negative samples
# app = NetVLADRenderer()

# TTM_BASE = 'data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
# app = TimeMachineRender(TTM_BASE)

PTS_BASE = 'data_Akihiko_Torii/Pitssburg/'
app = PittsburgRenderer( PTS_BASE )
# Preloading image folder 000
# app.preload_all_images( folder_list=[0] )

# WALKS_BASE = './keezi_walks/'
# app = WalksRenderer( WALKS_BASE )

n_positives = nP #5
n_negatives = nN #10
#TODO: Make the per iterations positive samples and negative samples settable from here. possibly as Arguments
# to the constructor. Using nP nN define above.

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

    tensorflow_session.run([zero_op,zero_tf_cost,zero_fit_loss,zero_reg_loss,zero_pos_set_dev]) #set gradient_cummulator and cost_cummulator to zero

    mini_batch = MINI_BATCH_SIZE
    n_zero_tff_costs = 0 #Number of zero-costs in this batch
    veri_total = 0.0; veri_fit=0.0; veri_reg=0.0
    # accumulate gradient
    for i_minibatch in range(mini_batch):
        im_batch, label_batch = app.step(nP=n_positives, nN=n_negatives, return_gray=False,ENABLE_IMSHOW=ENABLE_IMSHOW)
        while im_batch is None: #if queue not sufficiently filled, try again
            im_batch, label_batch = app.step(nP=n_positives, nN=n_negatives, return_gray=False, ENABLE_IMSHOW=ENABLE_IMSHOW)

        im_batch_normalized = normalize_batch( im_batch )

        #remember to set tf_x to 16,240,320,1 if using grays or to 16,240,320,3 if using 3-channels

        #vgg_obj.initial_t is for the loopy-tensorflow (tf.while_loop)
        feed_dict = {tf_x : im_batch_normalized,\
                     is_training:True,\
                     vgg_obj.initial_t: 0
                    }



        # print 'tf.run()', i_minibatch, im_batch.shape, im_batch_normalized.shape
        # tff_cost, tff_word, _grad_ = tensorflow_session.run( [tf_cost, tf_vlad_word, accum_op], feed_dict=feed_dict)
        # _dis_q_P, _dis_q_N, _cost = verify_cost( tff_word, nP, nN, margin )
        # print tff_cost, _cost
        # tff_cost, _grad_, tff_cc_cost, regloss = tensorflow_session.run( [tf_cost, accum_op, accum_cc_cost_op, regularization_loss], feed_dict=feed_dict)
        tff_cost, tff_fit, tff_regloss, tff_pos_set_dev, tff_cu_cost, tff_cu_fit, tff_cu_regloss, tff_cu_dev, _grad_, tff_dot_q_P, tff_dot_q_N = tensorflow_session.run( [tf_cost, fitting_loss, regularization_loss,  pos_set_dev, accum_tf_cost, accum_fit_loss, accum_reg_loss, accum_pos_set_dev, accum_op, vgg_obj.dot_q_P, vgg_obj.dot_q_N ], feed_dict=feed_dict )
        # veri_total += tff_cost
        # veri_fit   += tff_fit
        # veri_reg   += tff_regloss
        # print 'done tf.run()', i_minibatch
        if tff_fit <= 0.001:
            n_zero_tff_costs = n_zero_tff_costs + 1

        if i_minibatch >= 0:
            # print '%3d Pos' %(tf_iteration), tff_dot_q_P
            # print '%3d Neg' %(tf_iteration), tff_dot_q_N
            # plt.subplot( 4,6, i_minibatch+1)
            # plt.plot( np.ones(tff_dot_q_P.shape[0])*tf_iteration, tff_dot_q_P, 'g+' )
            # plt.plot( np.ones(tff_dot_q_N.shape[0])*tf_iteration, tff_dot_q_N, 'r.' )

            plt_pos_writer.write( '%d %d %s\n' %(tf_iteration,i_minibatch, str(tff_dot_q_P).replace('\n', ' ')[1:-1]) )
            plt_neg_writer.write( '%d %d %s\n' %(tf_iteration,i_minibatch, str(tff_dot_q_N).replace('\n', ' ')[1:-1]) )

        # plt.pause( 0.05 )

        print tcolor.OKBLUE, '%4.3f' %(tff_fit), tcolor.ENDC,
        print  tcolor.UNDERLINE, '(%4.3f)' %(tff_pos_set_dev), tcolor.ENDC,
    print

    cur_lr = get_learning_rate(tf_iteration, 0.0001)
    _, summary_exec,_ = tensorflow_session.run( [train_step,summary_op,tf_batch_success_ratio], feed_dict={tf_lr: cur_lr, tf_batch_success_ratio:n_zero_tff_costs } )

    # print '%3d(%8.2fms) : cost=%-8.3f cc_cost=%-8.3f fit_loss=%-8.6f reg_loss=%-8.3f n_zero_costs=%d/%d' %(tf_iteration, 1000.*(time.time() - startTime), mbatch_total_cost, tff_cc_cost, (tff_cc_cost-regloss*mini_batch), regloss*mini_batch, n_zero_tff_costs, mini_batch)
    elpTime = 1000.*(time.time() - startTime)
    print '%3d(%8.2fms) : total=%-8.3f fit_loss=%-8.3f dev=%-8.3f reg_loss=%-8.3f n_zeros=%d/%d) ' %(tf_iteration,elpTime, tff_cu_cost, tff_cu_fit, tff_cu_dev, tff_cu_regloss, n_zero_tff_costs, mini_batch)
    # print '%3d(%8.2fms) : cost(t/f/r) : (%8.2f/%8.2f/%8.2f) ' %(tf_iteration,elpTime, veri_total, veri_fit, veri_reg)


    # Periodically Save Models
    if tf_iteration % PARAM_n_write_tf_model == 0 and tf_iteration > 0:
        print 'Snapshot model : ', PARAM_model_save_prefix, tf_iteration
        tensorflow_saver.save( tensorflow_session, PARAM_model_save_prefix, global_step=tf_iteration )

    if tf_iteration % PARAM_n_write_summary == 0:
        print 'Write Summary : ', PARAM_tensorboard_prefix
        summary_writer.add_summary( summary_exec, tf_iteration )






    tf_iteration = tf_iteration + 1
    # code.interact( local=locals() )
