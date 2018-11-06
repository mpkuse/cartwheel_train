"""
    Contains my implementation of custom losses / validation functions.
    Works with keras2.0 and tf1.11.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 6th Nov, 2018
"""

from keras import backend as K
from keras.engine.topology import Layer
import keras

import numpy as np
#import cv2
import code


def triplet_loss2_maker( nP, nN, epsilon=0.3 ):
    # As per the NetVLAD paper's words
    # def triplet_loss2( params ):
        # y_true, y_pred = params
    def triplet_loss2( y_true, y_pred ):
        """ Closed negative sample - farthest positive sample """
        assert( y_pred.shape[1] == 1+nP+nN )

        # y_pred.shape = shape=(?, 5, 512)
        q = y_pred[:,0:1,:]    # shape=(?, 1, 512)
        P = y_pred[:,1:1+nP,:] # shape=(?, nP, 512)
        N = y_pred[:,1+nP:,:]  # shape=(?, nN, 512)

        q_dot_P = keras.layers.dot( [q,P], axes=-1 )  # shape=(?, 1, nP)
        q_dot_N = keras.layers.dot( [q,N], axes=-1 )  # shape=(?, 1, nN)


        # epsilon = 0.3  # Your epsilon here

        d_nearest_positive_sample = K.max( q_dot_P, axis=-1, keepdims=True )
        S = q_dot_N - d_nearest_positive_sample + epsilon #difference between best +ve and all negatives.
        return K.sum( K.maximum( 0., S ), axis=-1 )

    return triplet_loss2



def allpair_hinge_loss_maker( nP, nN, epsilon=0.3 ):
    # def allpair_hinge_loss( params ):
        # y_true, y_pred = params
    def allpair_hinge_loss(y_true, y_pred):
        """ All pair loss """
        # nP = 3
        # nN = 2
        assert( y_pred.shape[1] == 1+nP+nN )

        # y_pred.shape = shape=(?, 5, 512)
        q = y_pred[:,0:1,:]    # shape=(?, 1, 512)
        P = y_pred[:,1:1+nP,:] # shape=(?, 2, 512)
        N = y_pred[:,1+nP:,:]  # shape=(?, 2, 512)

        q_dot_P = keras.layers.dot( [q,P], axes=-1 )  # shape=(?, 1, 2)
        q_dot_N = keras.layers.dot( [q,N], axes=-1 )  # shape=(?, 1, 2)

        # epsilon = 0.3  # Your epsilon here

        zeros = K.zeros((nP, nN), dtype='float32')
        ones_m = K.ones((nP,1), dtype='float32')
        ones_n = K.ones((nN,1), dtype='float32')


        _1m__qdotN_T = ones_m[None,:] * q_dot_N # 1m ( \delta^q_N )^T
        qdotP__1n_T = K.permute_dimensions( ones_n[None,:] * q_dot_P, [0,2,1] ) # ( \delta^q_P ) 1n^T
        _1m__1n_T = epsilon * ones_m[None,:] * K.permute_dimensions( ones_n[None,:], [0,2,1] ) # 1m 1n^T

        aux = _1m__qdotN_T - qdotP__1n_T + _1m__1n_T

        return K.sum( K.maximum(zeros, aux) , axis=[-1,-2] )
    return allpair_hinge_loss



def allpair_count_goodfit_maker( nP, nN, epsilon=0.3 ):
    # def allpair_count_goodfit( params ):
        # y_true, y_pred = params
    def allpair_count_goodfit(y_true, y_pred):
        # nP = 3
        # nN = 2
        assert( y_pred.shape[1] == 1+nP+nN )

        # y_pred.shape = shape=(?, 5, 512)
        q = y_pred[:,0:1,:]    # shape=(?, 1, 512)
        P = y_pred[:,1:1+nP,:] # shape=(?, 2, 512)
        N = y_pred[:,1+nP:,:]  # shape=(?, 2, 512)

        q_dot_P = keras.layers.dot( [q,P], axes=-1 )  # shape=(?, 1, 2)
        q_dot_N = keras.layers.dot( [q,N], axes=-1 )  # shape=(?, 1, 2)

        # epsilon = 0.3  # Your epsilon here

        zeros = K.zeros((nP, nN), dtype='float32')
        ones_m = K.ones((nP,1), dtype='float32')
        ones_n = K.ones((nN,1), dtype='float32')

        _1m__qdotN_T = ones_m[None,:] * q_dot_N # 1m ( \delta^q_N )^T
        qdotP__1n_T = K.permute_dimensions( ones_n[None,:] * q_dot_P, [0,2,1] ) # ( \delta^q_P ) 1n^T
        _1m__1n_T = epsilon * ones_m[None,:] * K.permute_dimensions( ones_n[None,:], [0,2,1] ) # 1m 1n^T

        aux = _1m__qdotN_T - qdotP__1n_T + _1m__1n_T

        return K.sum( K.cast( K.less_equal( aux , 0),  'float32' ), axis=[-1,-2] ) #number of pairs which satisfy out of total nP*nN pairs
    return allpair_count_goodfit


def positive_set_deviation_maker( nP, nN ):
    # def positive_set_deviation( params ):
        # y_true, y_pred = params
    def positive_set_deviation(y_true, y_pred):
        assert( y_pred.shape[1] == 1+nP+nN )

        # y_pred.shape = shape=(?, 5, 512)
        q = y_pred[:,0:1,:]    # shape=(?, 1, 512)
        P = y_pred[:,1:1+nP,:] # shape=(?, 2, 512)
        N = y_pred[:,1+nP:,:]  # shape=(?, 2, 512)

        q_dot_P = keras.layers.dot( [q,P], axes=-1 )  # shape=(?, 1, nP)
        q_dot_N = keras.layers.dot( [q,N], axes=-1 )  # shape=(?, 1, nN)

        p_std = K.std( q_dot_P, axis=[-1,-2] )
        return p_std
    return positive_set_deviation


def allpair_hinge_loss_with_positive_set_deviation_maker( nP, nN, epsilon=0.3, opt_lambda=1.0 ):
    # def allpair_hinge_loss_with_positive_set_deviation( params ):
        # y_true, y_pred = params
    def allpair_hinge_loss_with_positive_set_deviation(y_true, y_pred):
        """ All pair loss with positive set deviation"""
        # nP = 3
        # nN = 2
        assert( y_pred.shape[1] == 1+nP+nN )

        # y_pred.shape = shape=(?, 5, 512)
        q = y_pred[:,0:1,:]    # shape=(?, 1, 512)
        P = y_pred[:,1:1+nP,:] # shape=(?, 2, 512)
        N = y_pred[:,1+nP:,:]  # shape=(?, 2, 512)

        q_dot_P = keras.layers.dot( [q,P], axes=-1 )  # shape=(?, 1, 2)
        q_dot_N = keras.layers.dot( [q,N], axes=-1 )  # shape=(?, 1, 2)

        # epsilon = 0.3  # Your epsilon here

        zeros = K.zeros((nP, nN), dtype='float32')
        ones_m = K.ones((nP,1), dtype='float32')
        ones_n = K.ones((nN,1), dtype='float32')


        _1m__qdotN_T = ones_m[None,:] * q_dot_N # 1m ( \delta^q_N )^T
        qdotP__1n_T = K.permute_dimensions( ones_n[None,:] * q_dot_P, [0,2,1] ) # ( \delta^q_P ) 1n^T
        _1m__1n_T = epsilon * ones_m[None,:] * K.permute_dimensions( ones_n[None,:], [0,2,1] ) # 1m 1n^T

        aux = _1m__qdotN_T - qdotP__1n_T + _1m__1n_T

        p_std = K.std( q_dot_P, axis=[-1,-2] ) #positive_set_deviation term
        return K.sum( K.maximum(zeros, aux) , axis=[-1,-2] ) + opt_lambda * p_std

    return allpair_hinge_loss_with_positive_set_deviation

# Verify loss function
if __name__ == '__main__':
    np.random.seed(0)
    nP = 3
    nN = 2
    y_true = keras.layers.Input( shape=(nP+nN+1,7) )
    y_pred = keras.layers.Input( shape=(nP+nN+1,7) )


    w = keras.layers.Lambda( allpair_hinge_loss_with_positive_set_deviation_maker( nP=nP, nN=nN, epsilon=0.3, opt_lambda=1.0) )( [y_true, y_pred] )
    # w = keras.layers.Lambda( allpair_hinge_loss_maker( nP=nP, nN=nN, epsilon=0.1) )( [y_true, y_pred] )
    # w_c = keras.layers.Lambda( allpair_count_goodfit_maker( nP=nP, nN=nN, epsilon=0.1) )( [y_true, y_pred] )
    # w_c = keras.layers.Lambda( positive_set_deviation_maker( nP=nP, nN=nN, opt_lambda=.5) )( [y_true, y_pred] )
    model = keras.models.Model( inputs=[y_true,y_pred], outputs=[w] )

    model.summary()
    keras.utils.plot_model( model, show_shapes=True )

    a = np.zeros( (10,6,7) ) #. this doesn't appear in the loss function as my loss functions are weakly supervised. don't care, this is y_true
    b_ = np.zeros( (10,6,7) )

    b = b_[0,:,:]
    b = np.round( np.random.random( (6,7) ), 2)
    b = b / np.linalg.norm( b, axis=1, keepdims=True )
    # b[0,0] = (3./5)
    # b[0,6] = (4./5)
    #
    # b[1,0] = (3./5)
    # b[1,1] = (4./5)
    #
    # b[2,0] = (3./5)
    # b[2,3] = (4./5)
    #
    # b[2,0] = (3./5)
    # b[2,3] = (4./5)
    #
    # b[2,0] = (3./5)
    # b[2,3] = (4./5)

    # b[1:3,:] = np.round( np.random.random( (2,7)), 2 )
    # b[3:,:] = np.round( np.random.random( (2,7)), 2 )
    b_[0,:,:] = b
    b_[2,:,:] = b

    out = model.predict( [a,b_] )

    aux = np.array( [[-0.05192798, -0.00773406],
       [ 0.1755529 ,  0.21974683],
       [ 0.06959844,  0.11379236]])
    quit()
