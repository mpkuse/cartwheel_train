"""
    Inspect's a trained model from keras
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


if __name__ == '__main__':
    nP = 2
    nN = 2

    #------
    # Load data on RAM
    #------

    PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
    pr = PittsburgRenderer( PTS_BASE )

    a, _ = pr.step( nP=5, nN=5, return_gray=False, resize=None, apply_distortions=False, ENABLE_IMSHOW=False )

    input_img = keras.layers.Input( shape=(480, 640, 3 ) )
    # cnn = input_img
    cnn = make_vgg( input_img )

    out = NetVLADLayer(num_clusters = 16)( cnn )
    # code.interact( local=locals() )
    model = keras.models.Model( inputs=input_img, outputs=out )

    model.load_weights( 'model.keras/core_model.keras' )

    out = model.predict( a[:,:,:,:] )
    np.matmul( out[0:1,:], np.transpose( out[1:,:] ) )
