""" A Class for defining the ResNet model in Tensorflow. """
import tensorflow as tf
import tensorflow.contrib.slim as slim


import numpy as np
import cv2
import matplotlib.pyplot as plt
import TerminalColors
import code

class CartWheelFlow:
    def __init__(self, trainable_on_device ):
        """ Constructor basically defines the SGD and the variables """
        print 'CartWheelFlow constructor...define trainable variables'
        self.tcol = TerminalColors.bcolors


        # Define all required variables (nightmare :'( )
        with tf.device( trainable_on_device ):
            with tf.variable_scope( 'trainable_vars', reuse=None ):

                #top-level conv
                wc_top = tf.get_variable( 'wc_top', [7,7,3,64], initializer=tf.contrib.layers.xavier_initializer_conv2d() )
                bc_top = tf.get_variable( 'bc_top', [64], initializer=tf.constant_initializer(0.01) )
                with tf.variable_scope( 'bn', reuse=None ):
                    wc_top_beta  = tf.get_variable( 'wc_top_beta', [64], initializer=tf.constant_initializer(value=0.0) )
                    wc_top_gamma = tf.get_variable( 'wc_top_gamma', [64], initializer=tf.constant_initializer(value=1.0) )
                    wc_top_pop_mean  = tf.get_variable( 'wc_top_pop_mean', [64], initializer=tf.constant_initializer(value=0.0), trainable=False )
                    wc_top_pop_varn  = tf.get_variable( 'wc_top_pop_varn', [64], initializer=tf.constant_initializer(value=0.0), trainable=False )



                ## RES2
                with tf.variable_scope( 'res2a', reuse=None ):
                    self._define_resnet_unit_variables( 64, [64,64,256], [1,3,1], False )

                with tf.variable_scope( 'res2b', reuse=None ):
                    self._define_resnet_unit_variables( 256, [64,64,256], [1,3,1], True )

                with tf.variable_scope( 'res2c', reuse=None ):
                    self._define_resnet_unit_variables( 256, [64,64,256], [1,3,1], True )


                ## RES3
                with tf.variable_scope( 'res3a', reuse=None ):
                    self._define_resnet_unit_variables( 256, [128,128,512], [1,3,1], False )

                with tf.variable_scope( 'res3b', reuse=None ):
                    self._define_resnet_unit_variables( 512, [128,128,512], [1,3,1], True )

                with tf.variable_scope( 'res3c', reuse=None ):
                    self._define_resnet_unit_variables( 512, [128,128,512], [1,3,1], True )

                with tf.variable_scope( 'res3d', reuse=None ):
                    self._define_resnet_unit_variables( 512, [128,128,512], [1,3,1], True )

                ## RES4
                with tf.variable_scope( 'res4a', reuse=None ):
                    self._define_resnet_unit_variables( 512, [256,256,1024], [1,3,1], False )

                with tf.variable_scope( 'res4b', reuse=None ):
                    self._define_resnet_unit_variables( 1024, [256,256,1024], [1,3,1], True )

                with tf.variable_scope( 'res4c', reuse=None ):
                    self._define_resnet_unit_variables( 1024, [256,256,1024], [1,3,1], True )

                with tf.variable_scope( 'res4d', reuse=None ):
                    self._define_resnet_unit_variables( 1024, [256,256,1024], [1,3,1], True )

                with tf.variable_scope( 'res4e', reuse=None ):
                    self._define_resnet_unit_variables( 1024, [256,256,1024], [1,3,1], True )

                ## RES5
                with tf.variable_scope( 'res5a', reuse=None ):
                    self._define_resnet_unit_variables( 1024, [512,512,2048], [1,3,1], False )

                with tf.variable_scope( 'res5b', reuse=None ):
                    self._define_resnet_unit_variables( 2048, [512,512,2048], [1,3,1], True )

                with tf.variable_scope( 'res5c', reuse=None ):
                    self._define_resnet_unit_variables( 2048, [512,512,2048], [1,3,1], True )



                ## Fully Connected Layer
                # NOTE: This will make each layer identical, if need be in the
                # future this can be changed to have different structures for
                # prediction variables.
                with tf.variable_scope( 'fully_connected', reuse=None ):
                    for scope_str in ['ZF']:#['x', 'y', 'z', 'yaw']:
                        with tf.variable_scope( scope_str, reuse=None ):
                            w_fc1 = tf.get_variable( 'w_fc1', [2048, 1024], initializer=tf.contrib.layers.xavier_initializer()) #This `2048` better be un-hardcoded. Donno a way to have this number in the constructor
                            w_fc2 = tf.get_variable( 'w_fc2', [1024, 512], initializer=tf.contrib.layers.xavier_initializer())
                            w_fc3 = tf.get_variable( 'w_fc3', [512, 120], initializer=tf.contrib.layers.xavier_initializer())

                            #bias terms
                            b_fc1 = tf.get_variable( 'b_fc1', [ 1024], initializer=tf.constant_initializer(0.01) )
                            b_fc2 = tf.get_variable( 'b_fc2', [ 512], initializer=tf.constant_initializer(0.01) )
                            b_fc3 = tf.get_variable( 'b_fc3', [ 120], initializer=tf.constant_initializer(0.01) )

                            with tf.variable_scope( 'bn' ):
                                w_fc1_beta = tf.get_variable( 'w_fc1_beta', [1024], initializer=tf.constant_initializer(value=0.0) )
                                w_fc1_gamma = tf.get_variable( 'w_fc1_gamma', [1024], initializer=tf.constant_initializer(value=1.0) )
                                w_fc1_pop_mean = tf.get_variable( 'w_fc1_pop_mean', [1024], initializer=tf.constant_initializer(value=1.0), trainable=False )
                                w_fc1_pop_varn = tf.get_variable( 'w_fc1_pop_varn', [1024], initializer=tf.constant_initializer(value=1.0), trainable=False )

                                w_fc2_beta = tf.get_variable( 'w_fc2_beta', [512], initializer=tf.constant_initializer(value=0.0) )
                                w_fc2_gamma = tf.get_variable( 'w_fc2_gamma', [512], initializer=tf.constant_initializer(value=1.0) )
                                w_fc2_pop_mean = tf.get_variable( 'w_fc2_pop_mean', [512], initializer=tf.constant_initializer(value=1.0), trainable=False )
                                w_fc2_pop_varn = tf.get_variable( 'w_fc2_pop_varn', [512], initializer=tf.constant_initializer(value=1.0), trainable=False )





        # Place the towers on each of the GPUs and compute ops for
        # fwd_flow, avg_gradient and update_variables

        print 'Exit successfully, from CartWheelFlow constructor'



    def resnet50_inference(self, x, is_training):
        """ This function creates the computational graph and returns the op which give a
            prediction given an input batch x
        """

        print 'Define ResNet50'
        #TODO: Expect x to be individually normalized, ie. for each image in the batch, it has mean=0 and var=1
        #      batch normalize input (linear scale only)


        with tf.variable_scope( 'trainable_vars', reuse=True ):
            wc_top = tf.get_variable( 'wc_top', [7,7,3,64] )
            bc_top = tf.get_variable( 'bc_top', [64] )
            with tf.variable_scope( 'bn', reuse=True ):
                wc_top_beta  = tf.get_variable( 'wc_top_beta', [64] )
                wc_top_gamma = tf.get_variable( 'wc_top_gamma', [64] )
                wc_top_pop_mean  = tf.get_variable( 'wc_top_pop_mean', [64] )
                wc_top_pop_varn  = tf.get_variable( 'wc_top_pop_varn', [64] )



            conv1 = self._conv2d( x, wc_top, bc_top, pop_mean=wc_top_pop_mean, pop_varn=wc_top_pop_varn, is_training=is_training, W_beta=wc_top_beta, W_gamma=wc_top_gamma, strides=2 )
            with tf.device( '/cpu:0'):
                tf.summary.scalar( 'sparsity_conv1', tf.nn.zero_fraction(conv1) )
                tf.summary.histogram( 'hist_conv1', conv1 )
            conv1 = self._maxpool2d( conv1, k=2 )

            input_var = conv1

            ## RES2
            with tf.variable_scope( 'res2a', reuse=True ):
                conv_out = self.resnet_unit( input_var, 64, [64,64,256], [1,3,1], is_training=is_training, short_circuit=False )

            with tf.variable_scope( 'res2b', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 256, [64,64,256], [1,3,1], is_training=is_training, short_circuit=True )

            with tf.variable_scope( 'res2c', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 256, [64,64,256], [1,3,1], is_training=is_training, short_circuit=True )

                ## MAXPOOL
                with tf.device( '/cpu:0'):
                    tf.summary.scalar( 'sparsity_res2', tf.nn.zero_fraction(conv_out) )
                    tf.summary.histogram( 'hist_res2', conv_out )
                conv_out = self._maxpool2d( conv_out, k=2 )


            ## RES3
            with tf.variable_scope( 'res3a', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 256, [128,128,512], [1,3,1], is_training=is_training, short_circuit=False )

            with tf.variable_scope( 'res3b', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 512, [128,128,512], [1,3,1], is_training=is_training, short_circuit=True )

            with tf.variable_scope( 'res3c', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 512, [128,128,512], [1,3,1], is_training=is_training, short_circuit=True )

            with tf.variable_scope( 'res3d', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 512, [128,128,512], [1,3,1], is_training=is_training, short_circuit=True )

                ## MAXPOOL
                with tf.device( '/cpu:0'):
                    tf.summary.scalar( 'sparsity_res3', tf.nn.zero_fraction(conv_out) )
                    tf.summary.histogram( 'hist_res3', conv_out )
                conv_out = self._maxpool2d( conv_out, k=2 )


            ## RES4
            with tf.variable_scope( 'res4a', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 512, [256,256,1024], [1,3,1], is_training=is_training, short_circuit=False )

            with tf.variable_scope( 'res4b', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [256,256,1024], [1,3,1], is_training=is_training, short_circuit=True )

            with tf.variable_scope( 'res4c', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [256,256,1024], [1,3,1], is_training=is_training, short_circuit=True )

            with tf.variable_scope( 'res4d', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [256,256,1024], [1,3,1], is_training=is_training, short_circuit=True )

            with tf.variable_scope( 'res4e', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [256,256,1024], [1,3,1], is_training=is_training, short_circuit=True )

                ## MAXPOOL
                with tf.device( '/cpu:0'):
                    tf.summary.scalar( 'sparsity_res4', tf.nn.zero_fraction(conv_out) )
                    tf.summary.histogram( 'hist_res4', conv_out )
                conv_out = self._maxpool2d( conv_out, k=2 )


            ## RES5
            with tf.variable_scope( 'res5a', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 1024, [512,512,2048], [1,3,1], is_training=is_training, short_circuit=False )

            with tf.variable_scope( 'res5b', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 2048, [512,512,2048], [1,3,1], is_training=is_training, short_circuit=True )

            with tf.variable_scope( 'res5c', reuse=True ):
                conv_out = self.resnet_unit( conv_out, 2048, [512,512,2048], [1,3,1], is_training=is_training, short_circuit=True )

                ## MAXPOOL
                with tf.device( '/cpu:0'):
                    tf.summary.scalar( 'sparsity_res5', tf.nn.zero_fraction(conv_out) )
                    tf.summary.histogram( 'hist_res5', conv_out )
                conv_out = self._avgpool2d( conv_out, k1=8, k2=10 )


            # Reshape Activations
            print conv_out.get_shape().as_list()[1:]
            n_activations = np.prod( conv_out.get_shape().as_list()[1:] )
            fc = tf.reshape( conv_out, [-1, n_activations] )

            # Fully Connected Layers

            with tf.variable_scope( 'fully_connected', reuse=True):
                for scope_str in ['ZF']:#['x', 'y', 'z', 'yaw']:
                    with tf.variable_scope( scope_str, reuse=True ):
                        pred = self._define_fc( fc, n_activations, [1024, 512, 120], is_training=is_training )
                        # print pred



            return pred





    def _print_tensor_info( self, display_str, T ):
        print self.tcol.WARNING, display_str, T.name, T.get_shape().as_list(), self.tcol.ENDC


    def resnet_unit( self, input_tensor, n_inputs, n_intermediates, intermediate_filter_size, is_training, short_circuit=True ):
        """ Defines the net structure of resnet unit
                input_tensor : Input of the unit
                n_inputs : Number of input channels
                n_intermediates : An array of intermediate filter outputs (usually len of this array is 3)
                intermediate_filter_size : Same size as `n_intermediates`, gives kernel sizes (sually 1,3,1)
                short_circuit : True will directly connect input to the (+-elementwise). False will add in a convolution before adding

                returns : output of the unit. after addition and relu

        """
        print '<--->'
        a = n_inputs
        b = n_intermediates #note, b[2] will be # of output filters
        c = intermediate_filter_size

        self._print_tensor_info( 'Input Tensor', input_tensor)

        wc1 = tf.get_variable( 'wc1', [c[0],c[0],a,b[0] ] )
        wc2 = tf.get_variable( 'wc2', [c[1],c[1],b[0],b[1]] )
        wc3 = tf.get_variable( 'wc3', [c[2],c[2],b[1],b[2]] )

        # BN variables
        #BN Adopted from http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        with tf.variable_scope( 'bn', reuse=True ):
            wc1_bn_beta  = tf.get_variable( 'wc1_beta', [b[0]] )
            wc1_bn_gamma = tf.get_variable( 'wc1_gamma', [b[0]] )
            wc1_bn_pop_mean  = tf.get_variable( 'wc1_pop_mean', [b[0]] )
            wc1_bn_pop_varn = tf.get_variable( 'wc1_pop_varn', [b[0]] )

            wc2_bn_beta  = tf.get_variable( 'wc2_beta', [b[1]] )
            wc2_bn_gamma = tf.get_variable( 'wc2_gamma', [b[1]] )
            wc2_bn_pop_mean = tf.get_variable( 'wc2_pop_mean', [b[1]] )
            wc2_bn_pop_varn = tf.get_variable( 'wc2_pop_varn', [b[1]] )

            wc3_bn_beta  = tf.get_variable( 'wc3_beta', [b[2]] )
            wc3_bn_gamma = tf.get_variable( 'wc3_gamma', [b[2]] )
            wc3_bn_pop_mean  = tf.get_variable( 'wc3_pop_mean', [b[2]] )
            wc3_bn_pop_varn  = tf.get_variable( 'wc3_pop_varn', [b[2]] )


        self._print_tensor_info( 'Request Var', wc1 )
        self._print_tensor_info( 'Request Var', wc2 )
        self._print_tensor_info( 'Request Var', wc3 )


        conv_1 = self._conv2d_nobias( input_tensor, wc1, pop_mean=wc1_bn_pop_mean, pop_varn=wc1_bn_pop_varn, is_training=is_training, W_beta=wc1_bn_beta, W_gamma=wc1_bn_gamma )
        conv_2 = self._conv2d_nobias( conv_1, wc2,  pop_mean=wc2_bn_pop_mean, pop_varn=wc2_bn_pop_varn, is_training=is_training, W_beta=wc2_bn_beta, W_gamma=wc2_bn_gamma )
        conv_3 = self._conv2d_nobias( conv_2, wc3, pop_mean=wc3_bn_pop_mean, pop_varn=wc3_bn_pop_varn, is_training=is_training,  W_beta=wc3_bn_beta, W_gamma=wc3_bn_gamma, relu_unit=False )
        self._print_tensor_info( 'conv_1', conv_1 )
        self._print_tensor_info( 'conv_2', conv_2 )
        self._print_tensor_info( 'conv_3', conv_3 )

        if short_circuit==True: #direct skip connection (no conv on side)
            conv_out = tf.nn.relu( tf.add( conv_3, input_tensor ) )
        else: #side connection has convolution
            wc_side = tf.get_variable( 'wc1_side', [1,1,a,b[2]] )
            with tf.variable_scope( 'bn', reuse=True ):
                wc_side_bn_beta = tf.get_variable( 'wc_side_bn_beta', [b[2]] )
                wc_side_bn_gamma = tf.get_variable( 'wc_side_bn_gamma', [b[2]] )
                wc_side_bn_pop_mean = tf.get_variable( 'wc_side_pop_mean', [b[2]] )
                wc_side_bn_pop_varn = tf.get_variable( 'wc_side_pop_varn', [b[2]] )

            self._print_tensor_info( 'Request Var', wc_side )
            conv_side = self._conv2d_nobias( input_tensor, wc_side, pop_mean=wc_side_bn_pop_mean, pop_varn=wc_side_bn_pop_varn, is_training=is_training, W_beta=wc_side_bn_beta, W_gamma=wc_side_bn_gamma, relu_unit=False )
            conv_out = tf.nn.relu( tf.add( conv_3, conv_side ) )

        self._print_tensor_info( 'conv_out', conv_out )
        return conv_out


    def _define_resnet_unit_variables( self, n_inputs, n_intermediates, intermediate_filter_size, short_circuit=True ):
        """ Defines variables in a resnet unit
                n_inputs : Number of input channels
                n_intermediates : An array of intermediate filter outputs (usually len of this array is 3)
                intermediate_filter_size : Same size as `n_intermediates`, gives kernel sizes (sually 1,3,1)
                short_circuit : True will directly connect input to the (+-elementwise). False will add in a convolution before adding

        """
        a = n_inputs
        b = n_intermediates #note, b[2] will be # of output filters
        c = intermediate_filter_size
        wc1 = tf.get_variable( 'wc1', [c[0],c[0],a,b[0]], initializer=tf.contrib.layers.xavier_initializer_conv2d() )
        wc2 = tf.get_variable( 'wc2', [c[1],c[1],b[0],b[1]], initializer=tf.contrib.layers.xavier_initializer_conv2d() )
        wc3 = tf.get_variable( 'wc3', [c[2],c[2],b[1],b[2]], initializer=tf.contrib.layers.xavier_initializer_conv2d() )

        #BN Adopted from http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        with tf.variable_scope( 'bn', reuse=None ):
            wc1_bn_beta  = tf.get_variable( 'wc1_beta', [b[0]], initializer=tf.constant_initializer(value=0.0) )
            wc1_bn_gamma = tf.get_variable( 'wc1_gamma', [b[0]], initializer=tf.constant_initializer(value=1.0) )
            wc1_bn_pop_mean  = tf.get_variable( 'wc1_pop_mean', [b[0]], initializer=tf.constant_initializer(value=0.0), trainable=False )
            wc1_bn_pop_varn = tf.get_variable( 'wc1_pop_varn', [b[0]], initializer=tf.constant_initializer(value=0.0), trainable=False )


            wc2_bn_beta  = tf.get_variable( 'wc2_beta', [b[1]], initializer=tf.constant_initializer(value=0.0) )
            wc2_bn_gamma = tf.get_variable( 'wc2_gamma', [b[1]], initializer=tf.constant_initializer(value=1.0) )
            wc2_bn_pop_mean = tf.get_variable( 'wc2_pop_mean', [b[1]], initializer=tf.constant_initializer(value=0.0), trainable=False )
            wc2_bn_pop_varn = tf.get_variable( 'wc2_pop_varn', [b[1]], initializer=tf.constant_initializer(value=0.0), trainable=False )


            wc3_bn_beta  = tf.get_variable( 'wc3_beta', [b[2]], initializer=tf.constant_initializer(value=0.0) )
            wc3_bn_gamma = tf.get_variable( 'wc3_gamma', [b[2]], initializer=tf.constant_initializer(value=1.0) )
            wc3_bn_pop_mean  = tf.get_variable( 'wc3_pop_mean', [b[2]], initializer=tf.constant_initializer(value=0.0), trainable=False )
            wc3_bn_pop_varn  = tf.get_variable( 'wc3_pop_varn', [b[2]], initializer=tf.constant_initializer(value=0.0), trainable=False )

        if short_circuit == False:
            wc_side = tf.get_variable( 'wc1_side', [1,1,a,b[2]], initializer=tf.contrib.layers.xavier_initializer_conv2d() )
            with tf.variable_scope( 'bn', reuse=None ):
                wc_side_bn_beta = tf.get_variable( 'wc_side_bn_beta', [b[2]], initializer=tf.constant_initializer(value=0.0) )
                wc_side_bn_gamma = tf.get_variable( 'wc_side_bn_gamma', [b[2]], initializer=tf.constant_initializer(value=1.0) )
                wc_side_bn_pop_mean = tf.get_variable( 'wc_side_pop_mean', [b[2]], initializer=tf.constant_initializer(value=0.0), trainable=False )
                wc_side_bn_pop_varn = tf.get_variable( 'wc_side_pop_varn', [b[2]], initializer=tf.constant_initializer(value=0.0), trainable=False )


    def _BN_fc( self, fc, w_beta, w_gamma, w_pop_mean, w_pop_varn, is_training=True ):
        """
            Does batch-normalization for fully connected layer
                fc: Output after matrix multiply. ie. before relu
                w_beta, w_gamma, w_pop_mean, w_pop_varn : variables for BN
                is_training  phase    """
        if is_training == True: #Phase : Training
            with tf.variable_scope( 'bn' ):
                # compute batch_mean
                batch_mean, batch_var = tf.nn.moments( fc, [0], name='moments' )

                # update population_mean
                decay = 0.999
                train_mean = tf.assign(w_pop_mean, w_pop_mean * decay + batch_mean * (1.0 - decay))
                train_var = tf.assign(w_pop_varn, w_pop_varn * decay + batch_var * (1.0 - decay))

                with tf.control_dependencies( [train_mean, train_var] ):
                    normed_x = tf.nn.batch_normalization( fc, batch_mean, batch_var, w_beta, w_gamma, 1E-3, name='apply_moments_training')
        else : #Phase : Testing
            with tf.variable_scope( 'bn' ):
                normed_x = tf.nn.batch_normalization( fc, w_pop_mean, w_pop_varn, w_beta, w_gamma, 1E-3, name='apply_moments_testing')

        return normed_x


    def _define_fc( self, fc, n_input, interim_input_dim, is_training ):
        """ Define a fully connected layer
                fc : the reshaped array
                n_input : number of inputs
                interim_input_dim : array of intermediate data dims

                Note that this assume, the context is already in correct scope
        """

        a = n_input
        b = interim_input_dim
        w_fc1 = tf.get_variable( 'w_fc1', [a, b[0]])
        w_fc2 = tf.get_variable( 'w_fc2', [b[0], b[1]])
        w_fc3 = tf.get_variable( 'w_fc3', [b[1], b[2]])

        b_fc1 = tf.get_variable( 'b_fc1', [ b[0]])
        b_fc2 = tf.get_variable( 'b_fc2', [ b[1]])
        b_fc3 = tf.get_variable( 'b_fc3', [ b[2]])

        # Get BN variables
        with tf.variable_scope( 'bn' ):
            w_fc1_beta = tf.get_variable( 'w_fc1_beta', [b[0]] )
            w_fc1_gamma = tf.get_variable( 'w_fc1_gamma', [b[0]] )
            w_fc1_pop_mean = tf.get_variable( 'w_fc1_pop_mean', [b[0]] )
            w_fc1_pop_varn = tf.get_variable( 'w_fc1_pop_varn', [b[0]] )

            w_fc2_beta = tf.get_variable( 'w_fc2_beta', [b[1]] )
            w_fc2_gamma = tf.get_variable( 'w_fc2_gamma', [b[1]] )
            w_fc2_pop_mean = tf.get_variable( 'w_fc2_pop_mean', [b[1]] )
            w_fc2_pop_varn = tf.get_variable( 'w_fc2_pop_varn', [b[1]] )



        fc1_pre = tf.add( tf.matmul( fc, w_fc1 ), b_fc1 )
        fc1_pre = self._BN_fc( fc1_pre, w_fc1_beta, w_fc1_gamma, w_fc1_pop_mean, w_fc1_pop_varn, is_training=is_training)
        fc1 = tf.nn.relu( fc1_pre )

        fc2_pre = tf.add( tf.matmul( fc1, w_fc2 ), b_fc2 )
        fc2_pre = self._BN_fc( fc2_pre, w_fc2_beta, w_fc2_gamma, w_fc2_pop_mean, w_fc2_pop_varn, is_training=is_training)
        fc2 = tf.nn.relu( fc2_pre )



        fc3 = tf.mul( 1., tf.add( tf.matmul( fc2, w_fc3 ), b_fc3, name='stacked_fc_out' ) )
        # tf.summary.histogram( 'a_input', fc )
        # tf.summary.histogram( 'fc1', fc1_pre )
        # tf.summary.histogram( 'fc2', fc2_pre )
        # tf.summary.histogram( 'fc3', fc3_pre )
        # tf.summary.histogram( 'fc4', fc4 )
        tf.summary.scalar( 'fc1_non_zeros', tf.nn.zero_fraction(fc1))
        tf.summary.scalar( 'fc2_non_zeros', tf.nn.zero_fraction(fc2))

        # tf.summary.histogram( 'w_fc1', w_fc1 )
        # tf.summary.histogram( 'w_fc2', w_fc2 )
        # tf.summary.histogram( 'w_fc3', w_fc3 )
        # tf.summary.histogram( 'w_fc4', w_fc4 )
        return fc3




    # Create some wrappers for simplicity
    def _conv2d(self, x, W, b, pop_mean, pop_varn, is_training, W_beta=None, W_gamma=None, strides=1):
        # Conv2D wrapper, with bias and relu activation

        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)

        if is_training == True: #Phase : Training
            with tf.variable_scope( 'bn' ):
                # compute batch_mean
                batch_mean, batch_var = tf.nn.moments( x, [0,1,2], name='moments' )

                # update population_mean
                decay = 0.999
                train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1.0 - decay))
                train_var = tf.assign(pop_varn, pop_varn * decay + batch_var * (1.0 - decay))

                with tf.control_dependencies( [train_mean, train_var] ):
                    normed_x = tf.nn.batch_normalization( x, batch_mean, batch_var, W_beta, W_gamma, 1E-3, name='apply_moments_training')
        else : #Phase : Testing
            with tf.variable_scope( 'bn' ):
                normed_x = tf.nn.batch_normalization( x, pop_mean, pop_varn, W_beta, W_gamma, 1E-3, name='apply_moments_testing')


        # NORMPROP
        # return (tf.nn.relu(x) - 0.039894228) / 0.58381937
        # return tf.nn.relu(x)
        return tf.nn.relu(normed_x)


    # Create some wrappers for simplicity
    def _conv2d_nobias(self, x, W, pop_mean, pop_varn, is_training, W_beta=None, W_gamma=None, strides=1, relu_unit=True, ):
        # Conv2D wrapper, with bias and relu activation

        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

        # NORMPROP
        # return (tf.nn.relu(x) - 0.039894228) / 0.58381937

        # Batch-Norm (BN)

        #if training then compute batch mean, update pop_mean, do normalization with batch mean
        #if testing, then do notmalization with pop_mean

        if is_training == True: #Phase : Training
            with tf.variable_scope( 'bn' ):
                # compute batch_mean
                batch_mean, batch_var = tf.nn.moments( x, [0,1,2], name='moments' )

                # update population_mean
                decay = 0.999
                train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1.0 - decay))
                train_var = tf.assign(pop_varn, pop_varn * decay + batch_var * (1.0 - decay))

                with tf.control_dependencies( [train_mean, train_var] ):
                    normed_x = tf.nn.batch_normalization( x, batch_mean, batch_var, W_beta, W_gamma, 1E-3, name='apply_moments_training')
        else : #Phase : Testing
            with tf.variable_scope( 'bn' ):
                normed_x = tf.nn.batch_normalization( x, pop_mean, pop_varn, W_beta, W_gamma, 1E-3, name='apply_moments_testing')



        if relu_unit == True:
            return tf.nn.relu(normed_x)
            # return tf.nn.relu(x)
        else:
            # NOTE : No RELU
            return normed_x
            # return x


    def _maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        pool_out = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')
        # NORMPROP
        # return (pool_out - 1.4850) / 0.7010
        return pool_out


    def _avgpool2d(self, x, k1=2, k2=2):
        # MaxPool2D wrapper
        pool_out = tf.nn.avg_pool(x, ksize=[1, k1, k2, 1], strides=[1, k1, k2, 1],
                              padding='SAME')
        # NORMPROP
        # return (pool_out - 1.4850) / 0.7010
        return pool_out


class VGGFlow:
    def __init__(self):
        x=0

    # vggnet16. is_training is a placeholder boolean
    def vgg16( self, inputs, is_training ):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],\
                          activation_fn=tf.nn.relu,\
                          #weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),\
                          weights_regularizer=slim.l2_regularizer(0.05),
                          normalizer_fn=slim.batch_norm, \
                          normalizer_params={'is_training':is_training, 'decay': 0.9, 'updates_collections': None}\
                          ):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [15, 20], scope='pool5')
            fc_dat = slim.flatten( net )

            pred_x   = slim.stack( fc_dat, slim.fully_connected, [128, 16, 1], scope='fcx' )
            pred_y   = slim.stack( fc_dat, slim.fully_connected, [128, 16, 1], scope='fcy' )
            pred_z   = slim.stack( fc_dat, slim.fully_connected, [128, 16, 1], scope='fcz' )
            pred_yaw = slim.stack( fc_dat, slim.fully_connected, [128, 16, 1], scope='fcyaw' )

            # net = slim.fully_connected(net, 4096, scope='fc6')
            # net = slim.fully_connected(net, 4096, scope='fc7')
            # net = slim.fully_connected(net, 4, activation_fn=None, scope='fc8')

            return pred_x, pred_y, pred_z, pred_yaw


## construct the VGG descriptor at 5 layers
class VGGDescriptor:
    def __init__(self, D=256, K=32, N=60*80, b=16):
        xdd = 0
        self._D = D#256
        self._K = K#32
        self._N = N#60*80
        self._b = b#16



    # vggnet16. is_training is a placeholder boolean
    def vgg16( self, inputs, is_training ):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],\
                          activation_fn=tf.nn.relu,\
                          weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),\
                          weights_regularizer=slim.l2_regularizer(0.001),
                          normalizer_fn=slim.batch_norm, \
                          normalizer_params={'is_training':is_training, 'decay': 0.9, 'updates_collections': None, 'scale': True}\
                          ):
            # tf.summary.histogram( 'xxxx_inputs', inputs )
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            # tf.summary.histogram( 'xxxx_blk1', net )
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            # tf.summary.histogram( 'xxxx_blk2', net )
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            # net = slim.repeat(net, 1, slim.conv2d, 256, [3, 3], scope='conv3')    #with relu and with BN
            net = slim.conv2d( net, 256, [3,3], activation_fn=None, scope='conv3' ) #w/o relu at the end. with BN. #TODO Possibly also remove BN from last one
            # tf.summary.histogram( 'xxxx_blk3', net )

            # net is now 16x60x80x256

            # MAX POOLING
            # TODO: To be replaced with NetVLAD-layer

            # ------ NetVLAD ------ #
            net = self.netvlad_layer( net ) #16x64x256, used 32 cluster instead of 64 for computation reason
            net = tf.nn.l2_normalize( net, dim=2, name='intra_normalization' )
            sh = tf.shape(net)
            net = tf.reshape( net, [sh[0], sh[1]*sh[2] ]) # retrns 16x64*256
            net = tf.nn.l2_normalize( net, dim=1, name='normalization' )
            return net
            # -------- ENDC ------- #


            # # ------ MaxPooling ----- #
            # net = slim.avg_pool2d(net, kernel_size=[10, 10], stride=10, scope='pool3') #after maxpool, net=16x6x8x256
            #
            # #intra normalize
            # net = tf.nn.l2_normalize( net, dim=3, name='intra_normalization' )
            # # net = tf.mul( tf.constant(1000.), net )
            #
            # sh = tf.shape(net)
            # net = tf.reshape( net, [sh[0], sh[1]*sh[2]*sh[3] ]) # retrns 16x12288
            #
            # #normalize
            # # net = tf.nn.l2_normalize( net, dim=1, name='l2_normalization' )
            # return net
            # # -------- ENDC --------- #



    ## Margined hinge loss. Distance is computed as euclidean distance.
    def svm_hinge_loss( self,tf_vlad_word, nP, nN, margin ):
        sp_q, sp_P, sp_N = tf.split_v( tf_vlad_word, [1,nP,nN], 0 )
        #sp_q=query; sp_P=definite_positives ; sp_N=definite_negatives
        #q:1x16k;   P:5x16k;    N:10x16k


        # distance between sp_q and each of sp_P
        one_a = tf.ones( [nP,1], tf.float32 )
        a_ = tf.sub( tf.matmul( one_a, sp_q ), sp_P ) #   (1 * q - P)**2  ==> (q-P)**2
        tf_dis_q_P = tf.reduce_mean( tf.mul( a_, a_ ), axis=1 ) #row-wise norm (L2)


        # distance between sp_q and each of sp_N
        one_b = tf.ones( [nN,1], tf.float32 )
        b_ = tf.sub( tf.matmul( one_b, sp_q ), sp_N ) #   (1 * q - P)**2  ==> (q-P)**2
        tf_dis_q_N = tf.reduce_mean( tf.mul( b_, b_ ), axis=1 ) #row-wise norm (L2)

        # SVM-hinge loss
        # max( tf_dis_q_P ): Farthest positive sample
        # min( tf_dis_q_N ): Nearest
        tf_margin = tf.constant(margin, name='margin')
        cost_ = tf.sub( tf.add( tf.reduce_max(tf_dis_q_P), tf_margin),  tf.reduce_min(tf_dis_q_N), name='svm_margin_loss' ) # max_i a - max_j b + m
        tf_cost = tf.maximum( cost_, tf.constant(0.0), name='hinge_loss' )

        return tf_cost



    ## log-sum-exp loss for every pair of (d_P, d_N). d_P \in Positive samples. d_N in negative samples
    def soft_ploss( self,tf_vlad_word, nP, nN, margin ):
        sp_q, sp_P, sp_N = tf.split_v( tf_vlad_word, [1,nP,nN], 0 )
        #sp_q=query; sp_P=definite_positives ; sp_N=definite_negatives
        #q:1x16k;   P:5x16k;    N:10x16k


        # distance between sp_q and each of sp_P
        one_a = tf.ones( [nP,1], tf.float32 )
        a_ = tf.sub( tf.matmul( one_a, sp_q ), sp_P ) #   (1 * q - P)**2  ==> (q-P)**2
        tf_dis_q_P = tf.reduce_sum( tf.mul( a_, a_ ), axis=1 ) #row-wise norm (L2)


        # distance between sp_q and each of sp_N
        one_b = tf.ones( [nN,1], tf.float32 )
        b_ = tf.sub( tf.matmul( one_b, sp_q ), sp_N ) #   (1 * q - P)**2  ==> (q-P)**2
        tf_dis_q_N = tf.reduce_sum( tf.mul( b_, b_ ), axis=1 ) #row-wise norm (L2)

        # form a 2D matrix for each pairwise distance difference
        # repeat positive_dis by nN times. and negative distance by nP time. (yes this is correct...you want to do the reverse)
        rep_P = tf.matmul( one_b, tf.expand_dims( tf_dis_q_P, 0 ) )
        rep_N = tf.matmul( one_a, tf.expand_dims( tf_dis_q_N, 0 ) )

        #pairwise difference of distances
        pdis_diff = rep_P - tf.transpose( rep_N ) + tf.constant(margin, name='margin')

        # logsumexp
        cost = tf.reduce_logsumexp( pdis_diff )
        hinged_cost = tf.maximum( cost, tf.constant(0.0), name='hinge_loss' )


        # self.cost = cost
        self.pdis_diff = pdis_diff
        # self.rep_P = rep_P
        # self.rep_N = rep_N
        # self.sp_q = sp_q
        # self.sp_P = sp_P
        # self.sp_N = sp_N
        self.tf_dis_q_P = tf_dis_q_P
        self.tf_dis_q_N = tf_dis_q_N
        return hinged_cost


    ## The words are l2_normalized. Comparison with dot product as against
    ## squared distance earlier with soft_ploss()
    def soft_angular_ploss( self,tf_vlad_word, nP, nN, margin ):
        sp_q, sp_P, sp_N = tf.split_v( tf_vlad_word, [1,nP,nN], 0 )
        #sp_q=query; sp_P=definite_positives ; sp_N=definite_negatives
        #q:1x16k;   P:5x16k;    N:10x16k

        # Dot Products : <q,P_i>  and <q,N_j>
        dot_q_P = tf.reduce_sum( tf.multiply( sp_q, sp_P ), axis=1 )
        dot_q_N = tf.reduce_sum( tf.multiply( sp_q, sp_N ), axis=1 )
        #TODO: Instead of using dot product use `acos( <q,P_i> )` as measure of
        # similarity. It will add a cosine stretching. Be careful to keep
        # cosine streatch not change the direction of similarity. Will have to
        # negate the angle. This is because, -1 should be mapped to a smaller
        # angle and +1 should be mapped to a larger angle.


        # Pairwise difference of measure of similarity
        one_a = tf.ones( [nP,1], tf.float32 )
        one_b = tf.ones( [nN,1], tf.float32 )

        rep_P = tf.matmul( one_b, tf.expand_dims( dot_q_P, 0 ) )
        rep_N = tf.matmul( one_a, tf.expand_dims( dot_q_N, 0 ) )

        # \forall (i,j) : <q, N_j> - <q,P_i>
        psimilarity_diff = -rep_P + tf.transpose( rep_N ) + tf.constant(margin, name='margin')


        # maximum. the pairwise distances range is (-2,2). 2 is added to cost to make it positive
        cost = tf.reduce_max( psimilarity_diff ) + tf.constant(2.0)
        return cost


    # computation of positive-set stddev.
    def positive_set_std_dev(  self,tf_vlad_word, nP, nN, scale_gamma=1.0 ):
        sp_P, sp_N = tf.split_v( tf_vlad_word, [1+nP,nN], 0 )
        #sp_P=similar cluster ; sp_N=outliers
        #P:6x16k;    N:10x16k

        mask = np.ones( [1+nP,1+nP], dtype=bool )
        B = np.tril_indices(1+nP) #lower triangular indices
        mask[B] = False

        mask_y = np.zeros( [1+nP, nN], dtype=bool )
        B_y = np.triu_indices(n=1+nP, m=nN) #n is for rows
        mask_y[B_y] = True

        XXt = tf.matmul( sp_P, tf.transpose(sp_P) )
        masked_XXt = tf.boolean_mask( XXt, mask ) #this is (15,1)

        XYt = tf.matmul( sp_P, tf.transpose(sp_N) )
        masked_XYt = tf.boolean_mask( XYt, mask_y ) #this is (45,1) 10+9+8+7+6+5



        # Computation of stddev of vector `masked_XXt`
        m = tf.reduce_mean(masked_XXt)
        stddev_n = tf.reduce_sum( tf.multiply( masked_XXt - m, masked_XXt - m ) )
        nTerms = (1+nP)*nP*0.5
        stddev_n = tf.sqrt( tf.multiply( 1./(nTerms-1), stddev_n ) )




        # Computation of stddev of vector `masked_XYt`
        m = tf.reduce_mean(masked_XYt)
        stddev_d = tf.reduce_sum( tf.multiply( masked_XYt - m, masked_XYt - m ) )
        nTerms =  (nN+nN-nP)*(nP+1)*0.5
        stddev_d = tf.sqrt( tf.multiply( 1./(nTerms-1), stddev_d ) )

        stddev = tf.div( stddev_n, stddev_d )
        stddev = tf.multiply( tf.constant( scale_gamma ), stddev )

        # self.p_sp_P = sp_P
        # self.p_sp_N = sp_N
        # self.p_XXt = XXt
        # self.p_masked_XXt = masked_XXt
        # self.p_stddev = stddev
        #
        # self.p_XYt = XYt
        # self.p_masked_XYt = masked_XYt
        return stddev






    def should_continue(self,t, *args):
        return t<self.time_steps

    def iteration(self,t,outputs_):
        D = self._D
        K = self._K
        N = self._N
        b = self._b
        #t^{th} cluster center. extract t^{th} row
        c_t = tf.slice( self.nl_c, [t,0], [1,D])

        #membership of each point wrt to t^{th} cluster. exract t^{th} col of sm
        sm_t = tf.slice( self.nl_sm, [0,t], [b*N,1] )


        diff = self.nl_Xd - c_t
        # ones_D = tf.constant( np.ones( (1,D), dtype='float32' ) )
        # diff_scaled = tf.multiply( diff, tf.matmul( sm_t, ones_D) )
        # tf.multiply has broadcasting, see test_tf_multiply.py
        diff_scaled = tf.multiply( sm_t, diff )


        vec_of_1_256 = []
        for bi in range(b):
            diff_slice = tf.slice( diff_scaled, [bi*N,0], [N, D] )
            vec_of_1_256.append( tf.reduce_sum( diff_slice, axis=0 ) )



        # outputs_ = outputs_.write(t, tf.segment_sum( diff, self.tf_e))
        outputs_ = outputs_.write( t, tf.stack(vec_of_1_256) )
        return t+1,outputs_


    ## NetVLAD layer
    ## Given a 16x60x80x256 input volume, out a clustered (K=64) ie. 16x(K*256) tensor
    def netvlad_layer( self, input_var ):

        D = self._D
        K = self._K
        N = self._N
        b = self._b


        #init netVLAD layer's trainable_variables
        with tf.variable_scope( 'netVLAD', reuse=None ):
            vlad_w = tf.get_variable( 'vlad_w', [1,1,D,K], initializer=tf.contrib.layers.xavier_initializer_conv2d())# 1x1xDxK
            vlad_b = tf.get_variable( 'vlad_b', [K], initializer=tf.contrib.layers.xavier_initializer()) #K
            vlad_c = tf.get_variable( 'vlad_c', [K,D], initializer=tf.contrib.layers.xavier_initializer()) #KxD




        ############# PART - I #################
        # 1x1 convolutions. input dim=D , output dim=K
        # after convolution, net must be 16x60x80xK
        netvlad_conv = tf.nn.bias_add( tf.nn.conv2d( input_var, vlad_w, strides=[1, 1, 1, 1], padding='VALID', name='netvlad_conv' ), vlad_b )


        # list all D-dim features (across images and across batches)
        sh = tf.shape(netvlad_conv)
        netvlad_conv_open = tf.reshape( netvlad_conv, [ sh[0]*sh[1]*sh[2], sh[3] ] ) #reshape. After reshape : (b*N)xK. N:=60x80


        # make segments (batchwise)
        e = []
        for ie in range(b):
            e.append( np.ones(N, dtype='int32')*ie )
        e = np.hstack( e )
        self.const_e = e
        # self.seg_e = tf.placeholder( dtype='int32', shape=(None) )
        self.tf_e = tf.constant( e )



        # softmax. output = (b*N) x K
        sm = tf.nn.softmax( netvlad_conv_open, name='netvlad_softmax')
        self.nl_netvlad_conv = netvlad_conv
        self.nl_input = input_var
        self.nl_sm = sm
        self.nl_c = vlad_c



        #TODO: Write a function to count number of items in each cluster. It is basically just axis=1 summation of `sm`
        # return sm, netvlad_conv, vlad_c #verified that this computation is correct.

        ############## Tensorflow Loopy ################
        sh = tf.shape(input_var)
        Xd = tf.reshape( input_var, [ sh[0]*sh[1]*sh[2], sh[3] ] ) #reshape. After reshape : (b*N)xK. N:=60x80
        self.nl_Xd = Xd
        self.time_steps = tf.shape( sm )[1]
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=self.time_steps)
        self.initial_t = tf.placeholder( dtype='int32')

        t, outputs = tf.while_loop( self.should_continue, self.iteration, [self.initial_t,initial_outputs] )
        outputs = tf.transpose( outputs.pack(), [1,0,2] )
        self.nl_outputs = outputs
        return outputs


        # ############# PART - II #################
        #
        # #C : 64 X 256 ie. KxD
        # #X : b*60*80 x 256 ie. bHWxD
        # each_c = tf.unstack( vlad_c, name='unstack_c'  ) #spits out 64 tensors. each of size 1x256
        # each_sm = tf.unstack( sm, num=K , axis=1, name='unstack_sm')
        # print len(each_sm)
        # code.interact( local=locals() )
        #
        #
        #
        #
        # #tmp
        # self.nl_summed = []
        # ones_D = tf.constant( np.ones( (1,D), dtype='float32' ) )
        # for k in range(5): #range( len(each_c) ):
        #     ff = Xd - each_c[k] #76800 x 256
        #     # sm_k = tf.expand_dims( each_sm[k], -1) #76800 x 1
        #
        #     # ff_scaled = tf.multiply( ff, tf.matmul( sm_k, ones_D) )
        #     ff_scaled = ff
        #
        #     summed = tf.segment_sum( ff_scaled, tf_e )
        #
        #     self.nl_summed.append( summed )
        #
        # self.nl_stacked = tf.pack( self.nl_summed )
        #
        # return summed
