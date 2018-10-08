"""
    Inspect's a trained model from keras
"""

# import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import keras
import code
import numpy as np
import time

import cv2

# Keras CUstom Implementation
from CustomNets import NetVLADLayer, make_vgg



# Data
from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer

# Custom Utils
from ColorLUT import ColorLUT
colx = ColorLUT()

if __name__ == '__main__':
    # WR_BASE = '/Bulk_Data/keezi_walks/'
    WR_BASE = '/media/mpkuse/Bulk_Data/keezi_walks/'
    wr = WalksRenderer( WR_BASE )
    for i in range(20):
        a,b = wr.step(nP=10, nN=10)
        print a.shape
        print b.shape
        cv2.waitKey(0)

if __name__ == '__1main__':
    nP = 3
    nN = 1

    #------
    # Load data on RAM
    #------

    # PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
    # pr = PittsburgRenderer( PTS_BASE )

    TTM_BASE = '/Bulk_Data/data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
    pr = TimeMachineRender( TTM_BASE )

    a, _ = pr.step( nP=nP, nN=nN, return_gray=False, resize=None, apply_distortions=False, ENABLE_IMSHOW=False )

    input_img = keras.layers.Input( shape=(480, 640, 3 ) )
    # cnn = input_img
    cnn = make_vgg( input_img )

    out = NetVLADLayer(num_clusters = 16)( cnn )
    # code.interact( local=locals() )
    model = keras.models.Model( inputs=input_img, outputs=out )

    # model.load_weights( 'model.keras/core_model_tryh.keras' )
    model.load_weights( 'model.keras/core_model_tryh.keras' )
    # model.load_weights( 'model.keras/core_model_tokyotm.keras' )

    # quit()
    start_t = time.time()
    for x in range(a.shape[0]):
        out, out_amap = model.predict( a[x:x+1,:,:,:] )
        out_amap_lut = colx.lut( out_amap[0,:,:] )
        cv2.imshow( 'im', a[x,:,:,:].astype('uint8'))
        cv2.imshow( 'out_amap_lut', out_amap_lut )
        cv2.waitKey(0)

    print 'predicted in %4.2fms' %( 1000. * (time.time() - start_t))
    out, out_amap = model.predict( a[:,:,:,:] )
    M = np.matmul( out[0:1,:], np.transpose( out[1:,:] ) )
    print M
