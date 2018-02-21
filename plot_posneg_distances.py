"""
    Plots the pos_0, neg_0 files. These file contain logging
    of dot product of q and P_i's (positive file). q and N_j's (negative file).
    These 2 files are written by train_netvlad.py when you train. usually
    written in tf.logs/A/ (ie. log folder of the training)

    eg. pos_0
        <itr> <batch_i> <5-vec>
            .
            .

    eg. neg_0
        <itr> <batch_i> <10-vec>

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 20th June, 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
#
import TerminalColors
tcolor = TerminalColors.bcolors()

# TODO Read as cmd-line arg
parser = argparse.ArgumentParser()
parser.add_argument( '-pos', "--positive_file", default='tf.logs/netvlad_k64_znormed/pos_0', help="Path of positive file. default:tf.logs/netvlad_k64_znormed/pos_0")
parser.add_argument( '-neg', '--negative_file',default='tf.logs/netvlad_k64_znormed/neg_0', help="Path of negative file. default:tf.logs/netvlad_k64_znormed/neg_0")
parser.add_argument( '-b', '--batch_id', type=int, default='-1', help='Batch id. for example 10 will only show the data for 10th batch of 24. default:ALL')
parser.add_argument( '-i', '--iteration', type=int, default='-1', help='Which iteration data to display. for example 23 will show data only for 23rd iteration. default:ALL')
parser.add_argument( '-s', '--skip', type=int, default='1', help='SKIP. Show data for every `s` iterations')
args = parser.parse_args()


FILE_pos = args.positive_file #'tf.logs/netvlad_k64_znormed/pos_0'
FILE_neg = args.negative_file #'tf.logs/netvlad_k64_znormed/neg_0'
print tcolor.HEADER, 'Read file : ', FILE_pos, tcolor.ENDC
print tcolor.HEADER, 'Read file : ', FILE_neg, tcolor.ENDC

P = np.loadtxt( FILE_pos )
N = np.loadtxt( FILE_neg )


# Switch for single plotting or multiple
PLOT_ = args.batch_id #-10 #negative denotes plot all. positive means plot single
if PLOT_ < 0 :
    print tcolor.HEADER, 'Plotting Mode : ALL', tcolor.ENDC
else:
    print tcolor.HEADER, 'Plotting Mode : %d' %(PLOT_), tcolor.ENDC


# Only show a particular iteration
ITER_ = args.iteration #-15 #negative means all iterations
if ITER_ < 0:
    print tcolor.HEADER, 'Iterations : ALL', tcolor.ENDC
else:
    print tcolor.HEADER, 'Iteration : ', ITER_, tcolor.ENDC

# Skip
SKIP = args.skip #1
print tcolor.HEADER, 'SKIP = %d' %(SKIP), tcolor.ENDC



for i in range( 0, P.shape[0], SKIP ):
    tf_iteration = int(P[i,0])

    batch_i = int(P[i,1])
    assert P[i,1] == N[i,1], "Files seem to inconsistent. At i=%d 2nd cols (batch numbers) do not match" %(i)


    if ITER_ > 0 and tf_iteration != ITER_:
        continue

    tff_dot_q_P = 1.01-P[i,2:]
    tff_dot_q_N = 1.01-N[i,2:]

    if PLOT_ < 0: # plot all
        # Multiple
        plt.subplot( 8, 3, batch_i+1 )
        plt.ylim( 0, 2 )
        plt.plot( np.ones(tff_dot_q_P.shape[0])*tf_iteration, tff_dot_q_P, 'g+' )
        plt.plot( np.ones(tff_dot_q_N.shape[0])*tf_iteration, tff_dot_q_N, 'r.' )
    else:
        # Single
        if batch_i == PLOT_:
            plt.ylim( 0, 2 )
            if i == 0:
                plt.plot( np.ones(tff_dot_q_P.shape[0])*tf_iteration, tff_dot_q_P, 'g+', label=r'$\langle \eta_q, \eta_{P_i} \rangle$' )
                plt.plot( np.ones(tff_dot_q_N.shape[0])*tf_iteration, tff_dot_q_N, 'r.', label=r'$\langle \eta_q, \eta_{N_i} \rangle$' )

            plt.plot( np.ones(tff_dot_q_P.shape[0])*tf_iteration, tff_dot_q_P, 'g+' )
            plt.plot( np.ones(tff_dot_q_N.shape[0])*tf_iteration, tff_dot_q_N, 'r.' )




plt.legend( loc='top right' , prop={'size': 28})
plt.xticks(fontsize=22, rotation=60)
plt.yticks(fontsize=22, rotation=90)
plt.xlabel( 'Learning Iterations', fontsize=22 )
plt.ylim( 0, 1)
plt.show()
