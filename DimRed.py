""" Learn Siamese Mapping
        Defines class DimRed to compute the mapping. This class will also
        have a siamese style loss function

        Roughly based on
        Hadsell, Raia, Sumit Chopra, and Yann LeCun. "Dimensionality
        reduction by learning an invariant mapping." Computer vision and pattern
        recognition, 2006 IEEE computer society conference on. Vol. 2. IEEE, 2006.

        Created : 24th Feb, 2017
        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


import numpy as np
import cv2
import matplotlib.pyplot as plt
import TerminalColors
import code


class DimRed():
    def __init__(self, n_input_dim=8192, n_intermediate_dim=512, n_output_dim=128):
        tcol = TerminalColors.bcolors()
        # Print info
        print tcol.HEADER, 'n_input_dim', tcol.ENDC, n_input_dim
        print tcol.HEADER, 'n_intermediate_dim', tcol.ENDC, n_intermediate_dim
        print tcol.HEADER, 'n_output_dim', tcol.ENDC, n_output_dim
        self.n_input_dim = n_input_dim
        self.n_intermediate_dim = n_intermediate_dim
        self.n_output_dim = n_output_dim
        self.n_8192 = n_input_dim
        self.n_512 = n_intermediate_dim
        self.n_128 = n_output_dim

        # Define trainable variables
        with tf.device( '/gpu:0'):
            with tf.variable_scope( 'fully_connected', reuse=None ):
                w_fc1 = tf.get_variable( 'w_fc1', [self.n_8192, self.n_512], initializer=tf.contrib.layers.xavier_initializer())
                b_fc1 = tf.get_variable( 'b_fc1', [self.n_512], initializer=tf.contrib.layers.xavier_initializer())

                w_fc2 = tf.get_variable( 'w_fc2', [self.n_512, self.n_128], initializer=tf.contrib.layers.xavier_initializer())
                b_fc2 = tf.get_variable( 'b_fc2', [self.n_128], initializer=tf.contrib.layers.xavier_initializer())

    def return_vars(self):
        with tf.variable_scope( 'fully_connected', reuse=True ):
            w_fc1 = tf.get_variable( 'w_fc1', [self.n_8192, self.n_512], initializer=tf.contrib.layers.xavier_initializer())
            b_fc1 = tf.get_variable( 'b_fc1', [self.n_512], initializer=tf.contrib.layers.xavier_initializer())

            w_fc2 = tf.get_variable( 'w_fc2', [self.n_512, self.n_128], initializer=tf.contrib.layers.xavier_initializer())
            b_fc2 = tf.get_variable( 'b_fc2', [self.n_128], initializer=tf.contrib.layers.xavier_initializer())
            return [w_fc1, b_fc1, w_fc2, b_fc2]


    ## Given a place holder `tf_x`. It is a bxD tensor, where b is the batch size
    ## and D is the input dimensionality. Returns a bxd tensor. d << D
    def fc( self,tf_x ):
        with tf.variable_scope( 'fully_connected', reuse=True ):
            w_fc1 = tf.get_variable( 'w_fc1', [self.n_8192, self.n_512] )
            b_fc1 = tf.get_variable( 'b_fc1', [self.n_512] )
            w_fc2 = tf.get_variable( 'w_fc2', [self.n_512, self.n_128] )
            b_fc2 = tf.get_variable( 'b_fc2', [self.n_128] )

        with tf.device( '/gpu:0'):
            c1_pre = tf.add( tf.matmul( tf_x, w_fc1 ), b_fc1 )
            fc1 = tf.nn.relu( c1_pre )
            c2_pre = tf.add( tf.matmul( fc1, w_fc2 ), b_fc2 ) #bx128


        return tf.nn.l2_normalize( c2_pre, dim=1) #each of 128-d vectors are normalized to unit length



    ## constrastive_loss
    ## reduced_x1 and reduced_x2 of size bxd
    ## Y of size bx1, indicating for each pair in batch about the nn-status
    def constrastive_loss( self, reduced_x1, reduced_x2, Y ):
        with tf.device( '/gpu:0'):
            Dw = 1.0 - tf.reduce_sum( tf.multiply( reduced_x1, reduced_x2 ), axis=1 ) #dot product
            # Dw = tf.cos( tf.multiply(2.0, Dw) )

            Ls = tf.multiply( Dw, Dw ) # Dw^2

            margin = 0.3 #the non neighbours only influence when they are nearer than `margin` in reduced space
            q = tf.maximum( 0.0, margin-Dw )
            Ld = tf.multiply( q, q)


            self.Dw = Dw
            self.Ls = Ls
            self.q = q
            self.Ld = Ld
            return tf.reduce_mean( tf.multiply( 1.0-Y, Ls) + tf.multiply( Y, Ld ) )

    def regularization_loss( self, reg_lambda ):
        with tf.device( '/cpu:0'):
            with tf.variable_scope( 'fully_connected', reuse=True ):
                w_fc1 = tf.get_variable( 'w_fc1', [self.n_8192, self.n_512] )
                b_fc1 = tf.get_variable( 'b_fc1', [self.n_512] )
                w_fc2 = tf.get_variable( 'w_fc2', [self.n_512, self.n_128] )
                b_fc2 = tf.get_variable( 'b_fc2', [self.n_128] )
            q = tf.nn.l2_loss( w_fc1 ) + tf.nn.l2_loss( w_fc2 )
            return tf.multiply( reg_lambda, q)
