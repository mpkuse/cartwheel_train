""" Separate Renderer and training. Have control of the event loop
        This function basically init the tensorflow and dequeue from the
        rendering class. This function holds the event loop. Training is done
        as a classification problem. The entire area is divided into grids
"""
import argparse
import time
import pickle

# Usual Math and Image Processing
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import caffe
import tensorflow as tf

from PandaRender import TrainRenderer
import CartWheelFlow as puf

import Noise
import censusTransform as ct
import SpaGrid

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


###### TENSORFLOW HELPERS ######
def define_softmax_loss( infer_logit_unnormed, class_label_hot ):

    #TODO: Regularization
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(infer_logit_unnormed, class_label_hot) )
    return cost

def define_l2_loss(  infer_op, label_x, label_y, label_z, label_yaw ):
    """ defines the l2-loss """
    loss_x = tf.reduce_mean( tf.square( tf.sub( infer_op[0], label_x ) ), name='loss_x' )
    loss_y = tf.reduce_mean( tf.square( tf.sub( infer_op[1], label_y ) ), name='loss_y' )
    loss_z = tf.reduce_mean( tf.square( tf.sub( infer_op[2], label_z ) ), name='loss_z' )
    loss_yaw = tf.reduce_mean( tf.square( tf.sub( infer_op[3], label_yaw ) ), name='loss_yaw' )

    xpy = tf.add( tf.mul( loss_x, 1.0 ), tf.mul( loss_y, 1.0 ), name='x__p__y')
    zpyaw = tf.add(tf.mul( loss_z, 1.0 ), tf.mul( loss_yaw, 0.5 ), name='z__p__yaw' )
    fitting_loss = tf.sqrt( tf.add(xpy,zpyaw,name='full_sum'), name='aggregated_loss'  )

    # regularization
    regularization_terms = []
    for vr in tf.trainable_variables():
        if vr.name.find( 'beta' ) <= 0 and vr.name.find('gamma') <= 0: #do not regularize beta and gamma
            #TODO : Selective regularization. Smaller regularization of early layers bigger regularization on later layers
            regularization_terms.append( tf.nn.l2_loss( vr ) )

    lambda_ = 0.01
    regularization_loss = tf.sqrt( tf.add_n( regularization_terms ) )
    regularization_loss = tf.mul( regularization_loss, lambda_ )
    tf.summary.scalar( 'loss/regularization', regularization_loss )
    tf.summary.scalar( 'loss/fitting', fitting_loss )

    return tf.add( fitting_loss, regularization_loss )


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

def _print_tensor_info(  display_str, T ):
        print tcolor.WARNING, display_str, T.name, T.device, T.get_shape().as_list(), tcolor.ENDC



# Given a square matrix, substract mean and divide std dev
def zNormalized( M ):
    M_mean = np.mean(M) # scalar
    M_std = np.std(M)
    if M_std < 0.0001 :
        return M

    M_statSquash = (M - M_mean)/(M_std+.0001)
    return M_statSquash

def _print_stats( txt, M ):
    print txt, ' : ', M.shape, M.dtype, np.mean(M), np.max(M), np.min(M)

def alexnet( x ):
    # with tf.variable_scope( 'trainables_vari', reuse=True ):
    with tf.device( '/cpu:0'):
        M = tf.get_variable( 'M', [57600,10] )

        Mx = tf.get_variable( 'Mx',   [10,1] )
        My = tf.get_variable( 'My',   [10,1] )
        Mz = tf.get_variable( 'Mz',   [10,1] )
        Myaw = tf.get_variable( 'Myaw', [10,1] )

    pool_out = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _print_tensor_info( 'pool_out', pool_out )

    fc_re = tf.reshape( pool_out, [-1, 57600] )
    _print_tensor_info( 'fc_re', fc_re )

    fc = tf.matmul( fc_re, M )

    fc_x = tf.matmul( fc, Mx )
    fc_y = tf.matmul( fc, My )
    fc_z = tf.matmul( fc, Mz )
    fc_yaw = tf.matmul( fc, Myaw )

    _print_tensor_info( 'fc', fc )
    _print_tensor_info( 'fc_x', fc_x )
    _print_tensor_info( 'fc_y', fc_y )
    _print_tensor_info( 'fc_z', fc_z )
    _print_tensor_info( 'fc_yaw', fc_yaw )

    return fc_x, fc_y, fc_z, fc_yaw


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




#functions for softdistribution

def gauss2D_pdf(x,y, mux, muy, sigma):
    num = (x - mux)**2 + (y-muy)**2
    return np.exp( -0.5/(sigma*sigma) * num  )


def get_soft_distribution( sg, X, Y, Z ):
    cen_array = []
    #TODO: the window (below) need to be reset as per Z.
    for x_off in range(-2,2+1):
        for y_off in range(-2,2+1):
            X_offset, Y_offset = sg.offset( X, Y, x_off, y_off )
            if X_offset is not None and Y_offset is not None:
                class_offset = sg.cord2Indx( X_offset, Y_offset )
                X_offset_cen, Y_offset_cen = sg.indx2Cord_centrum( class_offset )
                cen_array.append( (class_offset, X_offset_cen, Y_offset_cen))

                # print 'offset', class_offset, X_offset_cen, Y_offset_cen

    # Gaussian centered at X,Y with sigma= Z * np.tan(fov) / 3.0
    fov = 80
    sigma = Z * np.tan(fov/2) / 1.0
    probab = np.zeros(sg.n_classes)
    for (a_,b_,c_) in cen_array:
        # print ":",a_, gauss2D_pdf( b_, c_, X, Y, sigma)
        probab[a_] =  gauss2D_pdf( b_, c_, X, Y, sigma)
    probab = probab / np.sum( probab )

    return probab



###### END OF TF HELPERS ######


#
# Parse Commandline Arguments
PARAM_tensorboard_prefix, PARAM_n_write_summary, \
    PARAM_model_save_prefix, PARAM_n_write_tf_model, PARAM_model_restore = parse_cmd_args()
print tcolor.HEADER, 'tensorboard_prefix     : ', PARAM_tensorboard_prefix, tcolor.ENDC
print tcolor.HEADER, 'write_summary every    : ', PARAM_n_write_summary, 'iterations', tcolor.ENDC
print tcolor.HEADER, 'model_save_prefix      : ', PARAM_model_save_prefix, tcolor.ENDC
print tcolor.HEADER, 'write_tf_model every   : ', PARAM_n_write_tf_model, 'iterations', tcolor.ENDC

print tcolor.HEADER, 'model_restore          : ', PARAM_model_restore, tcolor.ENDC



# Init Renderer
app = TrainRenderer()
# app.run()




#
# Init Tensorflow
# #multigpu - have all the trainables on CPU

# This defines variables on CPU
puf_obj = puf.CartWheelFlow(trainable_on_device='/cpu:0')
# with tf.device( '/cpu:0' ):
#     with tf.variable_scope( 'trainables_vari', reuse=None ) as scope:
#         M = tf.get_variable( 'M', [57600,10], initializer=tf.contrib.layers.xavier_initializer() )
#
#         Mx = tf.get_variable( 'Mx',   [10,1], initializer=tf.contrib.layers.xavier_initializer() )
#         My = tf.get_variable( 'My',   [10,1], initializer=tf.contrib.layers.xavier_initializer() )
#         Mz = tf.get_variable( 'Mz',   [10,1], initializer=tf.contrib.layers.xavier_initializer() )
#         Myaw = tf.get_variable( 'Myaw', [10,1], initializer=tf.contrib.layers.xavier_initializer() )

# # #multigpu - SGD Optimizer on cpu
with tf.device( '/cpu:0' ):
    tf_learning_rate = tf.placeholder( 'float', shape=[], name='learning_rate' )
    tensorflow_optimizer = tf.train.AdamOptimizer( tf_learning_rate )

tf.summary.scalar('learning_rate', tf_learning_rate)
_print_trainable_variables()


# Define Deep Residual Nets
# #multigpu - have the infer computation on each gpu (with different batches)
tf_tower_cost = []
tf_tower_infer = []
tower_grad = []
tf_tower_ph_x = []
tf_tower_ph_label = []


for gpu_id in [0,1]:
    with tf.device( '/gpu:'+str(gpu_id) ):
        with tf.name_scope( 'tower_'+str(gpu_id) ):#, tf.variable_scope('trainables_vari', reuse=True):
            # have placeholder `x`, label_x, label_y, label_z, label_yaw
            tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
            tf_label = tf.placeholder( 'float',   [None,120], name='label_x')

            # infer op
            tf_infer_op = puf_obj.resnet50_inference(tf_x, is_training=True)  # Define these inference ops on all the GPUs
            # tf_infer_op = alexnet(tf_x)
            print tf_infer_op

            # Cost
            with tf.variable_scope( 'loss'):
                gpu_cost = define_softmax_loss( tf_infer_op, tf_label )

            # self._print_trainable_variables()


            # Gradient computation op
            # following grad variable contain a list of 2 elements each
            # ie. ( (grad_v0_gpu0,var0_gpu0),(grad_v1_gpu0,var1_gpu0) ....(grad_vN_gpu0,varN_gpu0) )
            tf_grad_compute = tensorflow_optimizer.compute_gradients( gpu_cost )


            # Make list of tower_cost, gradient, and placeholders
            tf_tower_cost.append( gpu_cost )
            tf_tower_infer.append( tf_infer_op )
            tower_grad.append( tf_grad_compute )

            tf_tower_ph_x.append(tf_x)
            tf_tower_ph_label.append(tf_label)



_print_trainable_variables()

tf.summary.scalar( 'cost_tower0', tf_tower_cost[0] )
tf.summary.scalar( 'cost_tower1', tf_tower_cost[1] )

cumm_cost = tf.add_n( tf_tower_cost, name='cumm_cost')
tf.summary.scalar( 'cumm_cost', cumm_cost )

# Average Gradients (gradient_gpu0 + gradient_gpu1 + ...)
with tf.device( '/gpu:0'):
    n_gpus = len( tower_grad )
    n_trainable_variables = len(tower_grad[0] )
    tf_avg_gradient = []

    for i in range( n_trainable_variables ): #loop over trainable variables
        t_var = tower_grad[0][i][1]

        t0_grad = tower_grad[0][i][0]
        t1_grad = tower_grad[1][i][0]

        # TODO : Generalized GPU version. Does not work for now.
        # ti_grad = [] #get Gradients from each gpus
        # for gpu_ in range( n_gpus ):
        #     ti_grad.append( tower_grad[gpu_][i][0] )
        #
        # grad_total = tf.add_n( ti_grad, name='gradient_adder' )

        grad_total = tf.add( t0_grad, t1_grad )

        frac = 1.0 / float(n_gpus)
        t_avg_grad = tf.mul( grad_total ,  frac, name='gradi_scaling' )

        tf_avg_gradient.append( (t_avg_grad, t_var) )


        with tf.variable_scope('trainables_summary'):
            if t_var.name.find( 'beta' ) <= 0 and t_var.name.find('gamma') <= 0:
                tf.summary.histogram( 'grad_'+t_var.name, t_avg_grad )
                tf.summary.histogram( 'var_'+t_var.name, t_var )
            # tf.summary.scalar( 'sparsity_grad'+t_var.name, tf.nn.zero_fraction(t_avg_grad))
            # tf.summary.scalar( 'sparsity_var'+t_var.name, tf.nn.zero_fraction(t_var))


with tf.device( '/cpu:0' ):
    # Have the averaged gradients from all GPUS here as arg for apply_grad()
    tensorflow_apply_grad = tensorflow_optimizer.apply_gradients( tf_avg_gradient )



# Fire up the TensorFlow-Session
# self.tensorflow_session = tf.Session( config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True) )
tensorflow_session = tf.Session( config=tf.ConfigProto(allow_soft_placement=True) )



# Tensorboard
summary_writer = tf.summary.FileWriter( PARAM_tensorboard_prefix, tensorflow_session.graph )
summary_op = tf.summary.merge_all()


# Saver
with tf.device( '/cpu:0' ):
    tensorflow_saver = tf.train.Saver()


# If PARAM_model_restore is none means init from scratch.
if PARAM_model_restore == None:
    print tcolor.OKGREEN,'global_variables_initializer() : xavier', tcolor.ENDC
    tensorflow_session.run( tf.global_variables_initializer() )
    tensorflow_iteration = 0
else:
    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver.restore( tensorflow_session, PARAM_model_restore )

    a__ = PARAM_model_restore.find('-')
    n__ = int(PARAM_model_restore[a__+1:])
    print tcolor.OKGREEN,'Restore Iteration : ', n__, tcolor.ENDC
    tensorflow_iteration =  n__#the number after `-`. eg. model-40000, 40000 will be set as iteration








tf.train.start_queue_runners(sess=tensorflow_session)

#
# with open( 'tf.logs/im_batch.pickle', 'rb' ) as handle:
#         im_batch = pickle.load(handle )
#
# with open( 'tf.logs/label_batch.pickle', 'rb' ) as handle:
#         label_batch = pickle.load(handle )
#


#
# Iterations
#itr = 0 #used for debug, can be removed
sg = SpaGrid.SpaGrid()

while True:
    startTime = time.time()

    batchsize=30
    im_batch, label_batch = app.step(batchsize=batchsize)
    # qA, qB = app.step(batchsize=30)



    # Option : a) can either send first few to GPU:0 and next few to GPU:1
    #          b) can pick random integers between 0,20 2 times
    r0 = range(0, int(batchsize/2))  #np.random.randint( 0, 20, 10 )
    r1 = range(int(batchsize/2), batchsize) #np.random.randint( 0, 20, 10 )
    # r0 = np.random.randint( 0, batchsize, int(batchsize/2) )
    # r1 = np.random.randint( 0, batchsize, int(batchsize/2) )


    #process batch
    # TODO : Possibly a function to do this
    class_hot = np.zeros( (batchsize,120) )
    for bt in range(batchsize):
        im_c = im_batch[bt,:,:,:]

        # if tensorflow_iteration % 3 == 0:
        #     im_noisy = Noise.noisy( 'gauss', im_c )
        # else:
        #     im_noisy = im_c
        # im_gry = np.mean( im_noisy, axis=2)
        # _print_stats( 'im_gry', im_gry)


        # cencusTR = ct.censusTransform( im_gry.astype('uint8') )
        # edges_out = cv2.Canny(cv2.blur(im_gry.astype('uint8'),(3,3)),100,200)
        # intTr = Noise.intensity_transform(im_gry.astype('uint8'))
        # intTr2 = Noise.intensity_transform(im_gry.astype('uint8'))

        # cv2.imwrite( 'dump_rendered/im_c_'+str(itr)+'_'+str(bt)+'.jpg', im_c.astype('uint8') )
        # # cv2.imwrite( 'dump_rendered/im_noisy_'+str(itr)+'_'+str(bt)+'.jpg', im_noisy.astype('uint8') )
        # cv2.imwrite( 'dump_rendered/intTr_'+str(itr)+'_'+str(bt)+'.jpg', intTr.astype('uint8') )
        # cv2.imwrite( 'dump_rendered/intTr2_'+str(itr)+'_'+str(bt)+'.jpg', intTr2.astype('uint8') )
        # cv2.imwrite( 'dump_rendered/cencusTR_'+str(itr)+'_'+str(bt)+'.jpg', cencusTR.astype('uint8') )


        im_batch[bt,:,:,0] = zNormalized( im_c[:,:,0] )
        im_batch[bt,:,:,1] = zNormalized( im_c[:,:,1] )
        im_batch[bt,:,:,2] = zNormalized( im_c[:,:,2] )
        # im_batch[bt,:,:,0] = zNormalized( intTr )
        # im_batch[bt,:,:,1] = zNormalized( intTr2 )
        # im_batch[bt,:,:,2] = zNormalized( cencusTR )
        # _print_stats( 'im_batch', im_batch)

        x_ = label_batch[bt,0]
        y_ = label_batch[bt,1]
        z_ = label_batch[bt,2]



        # # OLD and inflexible
        # CLASS = np.floor(x_/60) + np.floor(y_/60)*10  + 65
        # # CLASS_SOFTDIST =
        # class_hot[bt,CLASS] = 1.0
        # # print "CLASS",CLASS

        # NEW Way - Soft distribution
        probab = get_soft_distribution( sg, x_, y_, z_ )
        class_hot[bt,:] = probab


    feed = {\
    tf_tower_ph_x[0]:im_batch[r0,:,:,:],\
    tf_tower_ph_label[0]:class_hot[r0,0:120], \
    tf_tower_ph_x[1]:im_batch[r1,:,:,:],\
    tf_tower_ph_label[1]:class_hot[r1,0:120], \
    tf_learning_rate: get_learning_rate(tensorflow_iteration, 0.003)
    }

    # run
    _, pp,qq, summary_exec = tensorflow_session.run( [tensorflow_apply_grad, tf_tower_cost[0], tf_tower_cost[1], summary_op], feed_dict=feed )



    elapsedTimeMS = (time.time() - startTime)*1000.
    print '%3d(%2.2fms)  |  %2.2f %2.2f' %(tensorflow_iteration, elapsedTimeMS, pp, qq)


    # Periodically Save Models
    if tensorflow_iteration % PARAM_n_write_tf_model == 0 and tensorflow_iteration > 0:
        print 'Snapshot model : ', PARAM_model_save_prefix, tensorflow_iteration
        tensorflow_saver.save( tensorflow_session, PARAM_model_save_prefix, global_step=tensorflow_iteration )

    if tensorflow_iteration % PARAM_n_write_summary == 0 and tensorflow_iteration > 0:
        print 'Write Summary : ', PARAM_tensorboard_prefix
        summary_writer.add_summary( summary_exec, tensorflow_iteration )



    tensorflow_iteration = tensorflow_iteration+1
    app.taskMgr.step()
    # time.sleep(1)
