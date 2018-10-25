""" NetVLAD training. Using features from deeper VGG layers.

    - Input -- BN -- MobileNet -- NetVLAD
    - Deeper features (instead of currently using very shallow ones)
    - Verify my pairwise_loss also implement triplet loss
    - Keras sequence use

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 15th Oct, 2018
"""


from keras import backend as K
from keras.engine.topology import Layer
import keras

import code
import numpy as np
import cv2


# CustomNets
from CustomNets import NetVLADLayer
from CustomNets import dataload_, do_typical_data_aug
from CustomNets import make_from_mobilenet

from InteractiveLogger import InteractiveLogger


# Data
from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer

class WSequence(keras.utils.Sequence):
    def __init__(self, nP, nN, n_samples=500 ):
        # self.x, self.y = x_set, y_set
        self.D = dataload_( n_tokyoTimeMachine=n_samples, n_Pitssburg=-1, nP=nP, nN=nN )
        print 'dataload_ returned len(self.D)=', len(self.D), 'self.D[0].shape=', self.D[0].shape
        self.y = np.zeros( len(self.D) )

        # Data Augmentation
        # self.D = do_typical_data_aug( self.D )

        self.epoch = 0
        self.batch_size = 4
        self.refresh_data_after_n_epochs = 20
        self.n_samples = n_samples

    def __len__(self):
        return int(np.ceil(len(self.D) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.D[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array( batch_x ), np.array( batch_y )

       #TODO: Can return another number (sample_weight) for the sample. Which can be judge say by GMS matcher. If we see higher matches amongst +ve set ==> we have good positive samples,


    def on_epoch_end(self):
        N = self.refresh_data_after_n_epochs

        if self.epoch % N == 0 and self.epoch > 0 :
            print '[on_epoch_end] done %d epochs, so load new data\t' %(N), int_logr.dir()
            # Sample Data
            self.D = dataload_( n_tokyoTimeMachine=self.n_samples, n_Pitssburg=-1, nP=nP, nN=nN )

            if self.epoch > 400:
                # Data Augmentation
                self.D = do_typical_data_aug( self.D )

            print 'dataload_ returned len(self.D)=', len(self.D), 'self.D[0].shape=', self.D[0].shape
            self.y = np.zeros( len(self.D) )
            # modify data
        self.epoch += 1

        # TODO: start data augmentation after say 50 epochs. Don't do too heavy augmentation.




def triplet_loss( y_true, y_pred ):
# def triplet_loss( params ):
    # y_true, y_pred = params
    """ Closed negative sample - farthest positive sample """
    assert( y_pred.shape[1] == 1+nP+nN )

    # y_pred.shape = shape=(?, 5, 512)
    q = y_pred[:,0:1,:]    # shape=(?, 1, 512)
    P = y_pred[:,1:1+nP,:] # shape=(?, nP, 512)
    N = y_pred[:,1+nP:,:]  # shape=(?, nN, 512)

    q_dot_P = keras.layers.dot( [q,P], axes=-1 )  # shape=(?, 1, nP)
    q_dot_N = keras.layers.dot( [q,N], axes=-1 )  # shape=(?, 1, nN)


    epsilon = 0.3  # Your epsilon here
    d_nearest_negative_sample =  K.max( q_dot_N, axis=-1 )
    d_farthest_positive_sample = K.min( q_dot_P, axis=-1 )

    # case-A: d_nearest_negative_sample > d_farthest_positive_sample
    # Penalize this

    # case-B: d_nearest_negative_sample < d_farthest_positive_sample
    # This is desired, so this should have zero loss

    return K.maximum( 0., d_nearest_negative_sample - d_farthest_positive_sample + epsilon )


# As per the NetVLAD paper's words
def triplet_loss2( y_true, y_pred ):
# def triplet_loss2( params ):
    # y_true, y_pred = params
    """ Closed negative sample - farthest positive sample """
    assert( y_pred.shape[1] == 1+nP+nN )

    # y_pred.shape = shape=(?, 5, 512)
    q = y_pred[:,0:1,:]    # shape=(?, 1, 512)
    P = y_pred[:,1:1+nP,:] # shape=(?, nP, 512)
    N = y_pred[:,1+nP:,:]  # shape=(?, nN, 512)

    q_dot_P = keras.layers.dot( [q,P], axes=-1 )  # shape=(?, 1, nP)
    q_dot_N = keras.layers.dot( [q,N], axes=-1 )  # shape=(?, 1, nN)


    epsilon = 0.3  # Your epsilon here

    d_nearest_positive_sample = K.max( q_dot_P, axis=-1, keepdims=True )
    S = q_dot_N - d_nearest_positive_sample + epsilon #difference between best +ve and all negatives.
    return K.sum( K.maximum( 0., S ), axis=-1 )




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

    epsilon = 0.3  # Your epsilon here

    zeros = K.zeros((nP, nN), dtype='float32')
    ones_m = K.ones((nP,1), dtype='float32')
    ones_n = K.ones((nN,1), dtype='float32')


    _1m__qdotN_T = ones_m[None,:] * q_dot_N # 1m ( \delta^q_N )^T
    qdotP__1n_T = K.permute_dimensions( ones_n[None,:] * q_dot_P, [0,2,1] ) # ( \delta^q_P ) 1n^T
    _1m__1n_T = epsilon * ones_m[None,:] * K.permute_dimensions( ones_n[None,:], [0,2,1] ) # 1m 1n^T

    aux = _1m__qdotN_T - qdotP__1n_T + _1m__1n_T

    return K.sum( K.maximum(zeros, aux) , axis=[-1,-2] )


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

    epsilon = 0.3  # Your epsilon here

    zeros = K.zeros((nP, nN), dtype='float32')
    ones_m = K.ones((nP,1), dtype='float32')
    ones_n = K.ones((nN,1), dtype='float32')

    _1m__qdotN_T = ones_m[None,:] * q_dot_N # 1m ( \delta^q_N )^T
    qdotP__1n_T = K.permute_dimensions( ones_n[None,:] * q_dot_P, [0,2,1] ) # ( \delta^q_P ) 1n^T
    _1m__1n_T = epsilon * ones_m[None,:] * K.permute_dimensions( ones_n[None,:], [0,2,1] ) # 1m 1n^T

    aux = _1m__qdotN_T - qdotP__1n_T + _1m__1n_T

    return K.sum( K.cast( K.less_equal( aux , 0),  'float32' ), axis=[-1,-2] ) #number of pairs which satisfy out of total nP*nN pairs



# Verify loss function
if __name__ == '__1main__':
    np.random.seed(0)
    nP = 3
    nN = 2
    y_true = keras.layers.Input( shape=(6,7) )
    y_pred = keras.layers.Input( shape=(6,7) )

    # u = keras.layers.Lambda( allpair_hinge_loss )( [y_true, y_pred] )
    # v = keras.layers.Lambda( allpair_count_goodfit )( [y_true, y_pred] )
    w = keras.layers.Lambda( triplet_loss2 )( [y_true, y_pred] )
    model = keras.models.Model( inputs=[y_true,y_pred], outputs=w )

    model.summary()
    keras.utils.plot_model( model, show_shapes=True )

    a = np.zeros( (10,6,7) ) # don't care, this is y_true
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




# Training
if __name__ == '__main__':
    image_nrows = 240
    image_ncols = 320
    image_nchnl = 3
    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_allpairloss/' )
    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_tripletloss2/' )

    int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_quash_chnls_allpairloss/' )
    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_quash_chnls_tripletloss2/' )
    nP = 6
    nN = 6

    #--------------------------------------------------------------------------
    # Core Model Setup
    #--------------------------------------------------------------------------
    # Build
    input_img = keras.layers.Input( shape=(image_nrows, image_ncols, image_nchnl ) )
    cnn = make_from_mobilenet( input_img )
    cnn_dwn = keras.layers.Conv2D( 256, (1,1), padding='same', activation='relu' )( cnn )
    cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
    cnn_dwn = keras.layers.Conv2D( 32, (1,1), padding='same', activation='relu' )( cnn_dwn )
    cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
    cnn = cnn_dwn

    out, out_amap = NetVLADLayer(num_clusters = 16)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=out )

    # Plot
    model.summary()
    keras.utils.plot_model( model, to_file=int_logr.dir()+'/core.png', show_shapes=True )
    int_logr.add_file( 'model.json', model.to_json() )


    # Load Previous Weights
    # model.load_weights(  int_logr.dir() + '/core_model.keras' )




    #--------------------------------------------------------------------------
    # TimeDistributed
    #--------------------------------------------------------------------------
    t_input = keras.layers.Input( shape=(1+nP+nN, image_nrows, image_ncols, image_nchnl ) )
    t_out = keras.layers.TimeDistributed( model )( t_input )

    t_model = keras.models.Model( inputs=t_input, outputs=t_out )

    t_model.summary()
    keras.utils.plot_model( t_model, to_file=int_logr.dir()+'core_t.png', show_shapes=True )


    #--------------------------------------------------------------------------
    # Compile
    #--------------------------------------------------------------------------
    sgdopt = keras.optimizers.Adadelta(  )
    t_model.compile( loss=allpair_hinge_loss, optimizer=sgdopt, metrics=[allpair_count_goodfit] )
    # t_model.compile( loss=triplet_loss, optimizer=sgdopt, metrics=[allpair_count_goodfit] ) #not in use. TODO removal
    # t_model.compile( loss=triplet_loss2, optimizer=sgdopt, metrics=[allpair_count_goodfit] )
    # int_logr.fire_editor()

    import tensorflow as tf
    tb = tf.keras.callbacks.TensorBoard( log_dir=int_logr.dir() )


    t_model.fit_generator( generator=WSequence(nP, nN),
                            epochs=1500, verbose=1, initial_epoch=0,
                            validation_data = WSequence(nP, nN),
                            callbacks=[tb]
                         )
    print 'Save Final Model : ',  int_logr.dir() + '/core_model.keras'
    model.save( int_logr.dir() + '/core_model.keras' )
