""" NetVLAD training, slightly better model representation for
    cost computation.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 7th Oct, 2018
"""


# import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import keras
import code
import numpy as np

import cv2

# Keras CUstom Implementation
from CustomNets import NetVLADLayer
from CustomNets import make_vgg, make_upsampling_vgg, make_from_vgg19, make_from_vgg19_multiconvup
from CustomNets import do_augmentation, dataload_

from InteractiveLogger import InteractiveLogger


# Data
from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer

#TODO: Verify the computations of this function
#TODO: Also implement triplet loss and compare performance.
def pairwise_loss(y_true, y_pred):
    """ All pair loss """
    nP = 2
    nN = 2

    # y_pred.shape = shape=(?, 5, 512)
    q = y_pred[:,0:1,:]    # shape=(?, 1, 512)
    P = y_pred[:,1:1+nP,:] # shape=(?, 2, 512)
    N = y_pred[:,1+nP:,:]  # shape=(?, 2, 512)
    q_dot_P = keras.layers.dot( [q,P], axes=-1 )  # shape=(?, 1, 2)
    q_dot_N = keras.layers.dot( [q,N], axes=-1 )  # shape=(?, 1, 2)

    epsilon = 0.1  # Your epsilon here

    zeros = K.zeros((nP, nN), dtype='float32')
    ones_m = K.ones(nP, dtype='float32')
    ones_n = K.ones(nN, dtype='float32')
    aux = ones_m[None, :, None] * q_dot_N[:, None, :] \
          - q_dot_P[:, :, None] * ones_n[None, None, :] \
          + epsilon * ones_m[:, None] * ones_n[None, :]

    return K.maximum(zeros, aux)




def dataload_( n_tokyoTimeMachine, n_Pitssburg, nP, nN ):
    D = []
    if n_tokyoTimeMachine > 0 :
        TTM_BASE = '/Bulk_Data/data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
        pr = TimeMachineRender( TTM_BASE )
        for s in range(n_tokyoTimeMachine):
            a,_ = pr.step(nP=nP, nN=nN, return_gray=False, resize=(320,240), apply_distortions=False, ENABLE_IMSHOW=False)
            if s%100 == 0:
                print 'get a sample #', s
                print a.shape
            D.append( a )

    if n_Pitssburg > 0 :
        PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
        pr = PittsburgRenderer( PTS_BASE )
        for s in range(n_Pitssburg):
            a,_ = pr.step(nP=nP, nN=nN, return_gray=False, resize=(240,320), apply_distortions=False, ENABLE_IMSHOW=False)
            if s %100 == 0:
                print 'get a sample #', s
                print a.shape
            D.append( a )

    return D


# Test components
if __name__ == '__1main__':
    int_logr = InteractiveLogger( './models.keras/hu/' )



if __name__ == '__main__':
    nP = 2
    nN = 2

    int_logr = InteractiveLogger( './models.keras/model_learn_with_regul_multi_samplefit/' )



    #------------------------------------------------------------------------
    # Load data on RAM
    #------------------------------------------------------------------------
    D = dataload_( n_tokyoTimeMachine=2500, n_Pitssburg=-1, nP=nP, nN=nN )

    D = np.array( D )    #50 x (1+nP+nN) x 640 x 480 x 3
    # qu = D[:,0:1,:,:,:]  #50 x 1        x 640 x 480 x 3
    # pu = D[:,1:nP+1,:,:,:] #50 x nP       x 640 x 480 x 3
    # nu = D[:,1+nP:1+nP+nN:,:,:,:]  #50 x nN       x 640 x 480 x 3
    int_logr.add_linetext( 'Training Dataset shape='+str(D.shape) )


    #------------------------------------------------------------------------------
    # Augment Data
    #------------------------------------------------------------------------------

    # D = do_augmentation( D )
    int_logr.add_linetext( 'Training Dataset After DataAug shape='+str(D.shape) )


    image_nrows = D.shape[2] #480
    image_ncols = D.shape[3] #640
    image_nchnl = D.shape[4] #3
    print 'D.shape: ', D.shape



    #---------------------------------------------------------------------------
    # Setting Up core computation
    #---------------------------------------------------------------------------
    input_img = keras.layers.Input( shape=(image_nrows, image_ncols, image_nchnl ) )
    # cnn = input_img
    # cnn = make_upsampling_vgg( input_img )
    # cnn = make_from_vgg19( input_img, trainable=False )
    cnn = make_from_vgg19_multiconvup( input_img, trainable=True )


    # base_model = keras.applications.vgg19.VGG19(weights='imagenet')
    # base_model.summary()
    # keras.utils.plot_model( base_model, to_file='core.png', show_shapes=True )
    # quit()


    out, out_amap = NetVLADLayer(num_clusters = 16)( cnn )
    # code.interact( local=locals() )
    model = keras.models.Model( inputs=input_img, outputs=out )

    model.summary()
    keras.utils.plot_model( model, to_file=int_logr.dir()+'/core.png', show_shapes=True )
    int_logr.add_file( 'model.json', model.to_json() )

    # Set regularization for the layers
    r_l2=keras.regularizers.l2(0.0001)
    r_l1=keras.regularizers.l1(0.0001)
    for l in model.layers:
        print '---\n', l
        # l.trainable = False
        try:
            l.kernel_regularizer = r_l2
            print l.kernel_regularizer
        except:
            pass


    #--------------------------------------------------------------------------
    # TimeDistributed
    #--------------------------------------------------------------------------
    t_input = keras.layers.Input( shape=(1+nP+nN, image_nrows, image_ncols, image_nchnl ) )
    t_out = keras.layers.TimeDistributed( model )( t_input )

    t_model = keras.models.Model( inputs=t_input, outputs=t_out )

    t_model.summary()
    keras.utils.plot_model( t_model, to_file=int_logr.dir()+'core_t.png', show_shapes=True )


    # parallel_t_model = keras.utils.multi_gpu_model(t_model, gpus=2)


    #--------------------------------------------------------------------------
    # Compile
    #--------------------------------------------------------------------------
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.01, clipnorm=1.0)
    t_model.compile( loss=pairwise_loss, optimizer=rmsprop, metrics=[pairwise_loss] )
    int_logr.fire_editor()


    # callbacks : Tensorboard, lr, multigpu
    # Validation split and custom validation function
    import tensorflow as tf
    tb = tf.keras.callbacks.TensorBoard( log_dir=int_logr.dir() )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    checkpointer = keras.callbacks.ModelCheckpoint( filepath=int_logr.dir()+'/weights.{epoch:04d}.hdf5', verbose=1, period=10 )


    t_model.fit( x=D, y=np.zeros(D.shape[0]),
                epochs=20, batch_size=4, verbose=1, validation_split=0.1,initial_epoch=0,
                callbacks=[tb,checkpointer] )

    model.save( int_logr.dir() + '/core_model.keras' )


    for qi in range(1,10):
        D = dataload_( n_tokyoTimeMachine=1500, n_Pitssburg=-1, nP=nP, nN=nN )
        D = np.array( D )
        D = do_augmentation( D )

        t_model.fit( x=D, y=np.zeros(D.shape[0]),
                    epochs=200, batch_size=4, verbose=1, validation_split=0.1,initial_epoch=qi*20,
                    callbacks=[tb,checkpointer] )

        model.save( int_logr.dir() + '/core_model.keras' )
