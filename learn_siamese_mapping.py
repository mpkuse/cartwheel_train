""" Learn Siamese Mapping
        Takes input the dataset of netvlad descriptors for several images. The goal
        is to learn a mapping which still preserving the NN properties.
        Use make_db_netvlad.py script to generate a database.

        Roughly based on
        Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality
        reduction by learning an invariant mapping." Computer vision and pattern
        recognition, 2006 IEEE computer society conference on. Vol. 2. IEEE, 2006.

        Created : 24th Feb, 2017
        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
"""
import numpy as np
import cv2
from annoy import AnnoyIndex
import time
import code

import tensorflow as tf
import DimRed
import NeighbourFactory

#
import TerminalColors
tcolor = TerminalColors.bcolors()


# PARAM_DESCRIPTOR_DB = 'dim_red_training_dat_random.npz'
PARAM_DESCRIPTOR_DB = 'tf.logs/netvlad_k48/db2/vlad_words_db.npz'
PARAM_DIST_THRESH = 0.630 #to keep NN whose distances is less than this. 0.630 is equivalent to 0.80 in dot-product

PARAM_model_restore = None

sl = PARAM_DESCRIPTOR_DB.rfind( '/' )
PARAM_tensorboard_prefix = PARAM_DESCRIPTOR_DB[:sl] + '/siamese_dimred_slowlr/'
# PARAM_tensorboard_prefix = 'tf.logs/netvlad_k48/siamese_dimred/'
PARAM_model_save_prefix  = PARAM_tensorboard_prefix+'/model'

PARAM_net_intermediate_dim = 1024
PARAM_net_out_dim = 256

print tcolor.HEADER, 'NETVLAD_DESCRIPTOR_DB : ', PARAM_DESCRIPTOR_DB, tcolor.ENDC
print tcolor.HEADER, 'SIAMESE_DIMRED_MODEL_SAVE : ', PARAM_tensorboard_prefix, tcolor.ENDC
print tcolor.HEADER, 'SIAMESE_NET : NONE x',PARAM_net_intermediate_dim, 'x', PARAM_net_out_dim, tcolor.ENDC


# Load Training Data and Make KD-Tree
fty = NeighbourFactory.NeighbourFactory(PARAM_DESCRIPTOR_DB)

def learning_rate(base_rate, itr):
    if itr<50:
        return base_rate
    elif itr<100:
        return base_rate/2.0
    elif itr<200:
        return base_rate/4.0
    elif itr<300:
        return base_rate/8.0
    elif itr<500:
        return base_rate/12.0

#
# Init Tensorflow Learning
tf_x1 = tf.placeholder( 'float', [None,fty.M.shape[1]], name='x1' )
tf_x2 = tf.placeholder( 'float', [None,fty.M.shape[1]], name='x2' )
tf_Y  = tf.placeholder( 'float', [None], name='Y' ) #=1 if in neighbourhood. else 0

net = DimRed.DimRed(n_input_dim=fty.M.shape[1], n_intermediate_dim=PARAM_net_intermediate_dim, n_output_dim=PARAM_net_out_dim)
reduced_x1 = net.fc( tf_x1 )
reduced_x2 = net.fc( tf_x2 )
print tcolor.OKGREEN, 'Setup Siamese Network for dimensionality reduction', tcolor.ENDC

tf_fit_cost = net.constrastive_loss( reduced_x1, reduced_x2, tf_Y )
tf_reg_cost = net.regularization_loss( 0.00001 )
tf_cost =  tf_fit_cost + tf_reg_cost
tf.summary.scalar( 'tf_fit_cost', tf_fit_cost )
tf.summary.scalar( 'tf_reg_cost', tf_reg_cost )
tf.summary.scalar( 'tf_cost', tf_cost )

print tcolor.OKGREEN, 'Setup `constrastive_loss`', tcolor.ENDC


lr = tf.placeholder( tf.float32, shape=[] )
tf.summary.scalar( 'lr', lr )
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(tf_cost)
print tcolor.OKGREEN, 'Setup Stochastic Gradient Descent', tcolor.ENDC

#
# Fireup Tensorflow
tensorflow_session = tf.Session()
tensorflow_saver = tf.train.Saver()
if PARAM_model_restore == None:
    print tcolor.OKGREEN,'global_variables_initializer() : xavier', tcolor.ENDC
    tensorflow_session.run( tf.global_variables_initializer() )
else:
    print tcolor.OKGREEN,'Restore model from : ', PARAM_model_restore, tcolor.ENDC
    tensorflow_saver.restore( tensorflow_session, PARAM_model_restore )



#
# Summary Writer and Saver
print tcolor.OKGREEN, 'PARAM_tensorboard_prefix : ', PARAM_tensorboard_prefix, tcolor.ENDC
summary_writer = tf.summary.FileWriter( PARAM_tensorboard_prefix, tensorflow_session.graph )
summary_op = tf.summary.merge_all()

########################################
#
# Iterations
batch_size = 4000
n_conjoining = 20 #neighbours
n_nonjoining = 20 #non neighbours
print tcolor.OKGREEN, 'Start descent iterations with batchsize=%d, #conjoins=%d, #nonjoins=%d' %(batch_size, n_conjoining, n_nonjoining), tcolor.ENDC
for itr in range(10000):
    # print 'ITERATION ', itr
    # Make feed_dict
    startTime = time.time()
    ti=0
    X1 = np.zeros( (batch_size,fty.M.shape[1]) )
    X2 = np.zeros( (batch_size,fty.M.shape[1]) )
    Y  = np.zeros( batch_size )
    for u in np.random.randint(0, fty.M.shape[0], 2*batch_size ): #1000 query points
        if ti >= batch_size:
            # print 'batch full'
            break
        nei = fty.get_neighbours( u, 4*n_conjoining, PARAM_DIST_THRESH ) #This distance is sqrt(2(1-cos(u,v))) distance
        non_nei = fty.get_non_neighbours( u, 4*n_nonjoining )
        # print tcolor.OKGREEN, 'feasible neighbours of ', u, ": ", nei, tcolor.ENDC
        # print tcolor.FAIL,    'def non neighbours  of ', u, ": ", non_nei, tcolor.ENDC

        if len(nei) > (n_conjoining+1):
            for j in range( 1,(n_conjoining+1) ): #start from 1 as best neighbout of u will be u, so to skip this
                # print '%5d <---> %5d (ti=%3d); Y=0 (similar)' %( u, nei[j], ti)
                X1[ti,:] = fty.M[u,:]
                X2[ti,:] = fty.M[nei[j],:]
                Y[ti] = 0.0
                ti = ti + 1
            for j in range( n_nonjoining ):
                # print '%5d <---> %5d (ti=%3d); Y=1 (dis-similar)' %( u, non_nei[j],ti )
                X1[ti,:] = fty.M[u,:]
                X2[ti,:] = fty.M[non_nei[j],:]
                Y[ti] = 1.0
                ti = ti + 1
        # else:
            # print 'Skip Query u=',u

    time_assembly = time.time() - startTime
    # print 'feed_dict assembled in %4.2fs' %(time_assembly)


    # Run tensorflow 1 learning-iteration
    startTime = time.time()
    feed_dict = { tf_x1: X1,\
                  tf_x2: X2,\
                  tf_Y : Y,\
                  lr   : learning_rate(0.0005, itr)
                }
    _, summary_exec, tff_cost, tff_fit_cost, tff_reg_cost = tensorflow_session.run( [train_op, summary_op, tf_cost, tf_fit_cost, tf_reg_cost ], feed_dict=feed_dict )

    # proc = [reduced_x1,reduced_x2,net.Dw, net.Ls,net.q,  tf_cost]
    # tff_r1, tff_r2, tff_Dw, tff_Ls, tff_q, tff_cost = tensorflow_session.run( proc, feed_dict=feed_dict )

    time_descent = (time.time() - startTime)
    print 'tf%4d [%4.2fs/%4.2fs] total_cost=%8.5f fit_cost=%8.5f reg_cost=%8.5f ti=%5d' \
                            %(itr, time_assembly, time_descent, tff_cost, tff_fit_cost, tff_reg_cost, ti )

    # Summary Writing
    if itr % 5 == 0:
        print tcolor.WARNING, 'Write Summary : ', PARAM_tensorboard_prefix, tcolor.ENDC
        summary_writer.add_summary( summary_exec, itr )

    # Write model
    if itr % 100 == 0:
        print tcolor.WARNING, 'Write Model : ', PARAM_model_save_prefix, tcolor.ENDC
        tensorflow_saver.save( tensorflow_session, PARAM_model_save_prefix, global_step=itr )



    # code.interact(local=locals())

quit()



################################
# #
# # Dry Iterations
# feed_dict = { tf_x1: M[0:3,:],\
#               tf_x2: M[10:13,:],\
#               tf_Y : [0,0,0]
#             }
# proc = [reduced_x1,reduced_x2,net.Dw, net.Ls,net.q,  tf_cost]
# tff_r1, tff_r2, tff_Dw, tff_Ls, tff_q, tff_cost = tensorflow_session.run( proc, feed_dict=feed_dict )
#
# quit()
