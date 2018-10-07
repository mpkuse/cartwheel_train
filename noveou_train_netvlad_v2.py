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
from CustomNets import NetVLADLayer, make_vgg



# Data
from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer


def custom_loss(y_true, y_pred):
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
    # code.interact( local=locals() , banner='custom_loss')
    aux = ones_m[None, :, None] * q_dot_N[:, None, :] \
          - q_dot_P[:, :, None] * ones_n[None, None, :] \
          + epsilon * ones_m[:, None] * ones_n[None, :]

    return K.maximum(zeros, aux)


if __name__ == '__main__':
    nP = 2
    nN = 2

    #------------------------------------------------------------------------
    # Load data on RAM
    #------------------------------------------------------------------------
    PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
    pr = PittsburgRenderer( PTS_BASE )
    D = []
    for s in range(100):
        print 'get a sample #', s
        a,_ = pr.step(nP=nP, nN=nN, return_gray=False, resize=(240,320), apply_distortions=False, ENABLE_IMSHOW=False)
        print a.shape
        D.append( a )
        # for i in range( a.shape[0] ):
        #     print i
        #     cv2.imshow( 'win', a[i,:,:,:].astype('uint8'))
        #     cv2.waitKey(0)

    D = np.array( D )    #50 x (1+nP+nN) x 640 x 480 x 3
    # qu = D[:,0:1,:,:,:]  #50 x 1        x 640 x 480 x 3
    # pu = D[:,1:nP+1,:,:,:] #50 x nP       x 640 x 480 x 3
    # nu = D[:,1+nP:1+nP+nN:,:,:,:]  #50 x nN       x 640 x 480 x 3
    image_nrows = D.shape[2] #480
    image_ncols = D.shape[3] #640
    image_nchnl = D.shape[4] #3


    #---------------------------------------------------------------------------
    # Setting Up core computation
    #---------------------------------------------------------------------------
    input_img = keras.layers.Input( shape=(image_nrows, image_ncols, image_nchnl ) )
    # cnn = input_img
    cnn = make_vgg( input_img )

    out = NetVLADLayer(num_clusters = 16)( cnn )
    # code.interact( local=locals() )
    model = keras.models.Model( inputs=input_img, outputs=out )

    model.summary()
    keras.utils.plot_model( model, to_file='core.png', show_shapes=True )


    #--------------------------------------------------------------------------
    # TimeDistributed
    #--------------------------------------------------------------------------
    t_input = keras.layers.Input( shape=(1+nP+nN, image_nrows, image_ncols, image_nchnl ) )
    t_out = keras.layers.TimeDistributed( model )( t_input )

    t_model = keras.models.Model( inputs=t_input, outputs=t_out )

    t_model.summary()
    keras.utils.plot_model( t_model, to_file='core_t.png', show_shapes=True )


    #--------------------------------------------------------------------------
    # Compile
    #--------------------------------------------------------------------------
    t_model.compile( loss=custom_loss, optimizer='sgd' )

    t_model.fit( x=D, y=np.zeros(D.shape[0]),
                epochs=5, batch_size=20, verbose=2 )
