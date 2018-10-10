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
from CustomNets import NetVLADLayer
from CustomNets import make_vgg, make_upsampling_vgg




# Data
from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer

# Custom Utils
from ColorLUT import ColorLUT
colx = ColorLUT()

if __name__ == '__m1ain__':
    # WR_BASE = '/Bulk_Data/keezi_walks/'
    WR_BASE = '/media/mpkuse/Bulk_Data/keezi_walks/'
    wr = WalksRenderer( WR_BASE )
    for i in range(20):
        a,b = wr.step(nP=10, nN=10)
        print a.shape
        print b.shape
        cv2.waitKey(0)

if __name__ == '__mai1n__': # Use __seq__


    ##
    ## Load Data
    ##
    base = '/Bulk_Data/__seq_nap_out__/'
    seq = 'base-2'
    seq = 'tpt-park'
    seq = 'coffee-shop'
    seq = 'lsk-1'

    fname = base+'/'+seq+'/S_full_res.npy'
    print 'Load : ', fname
    S_full_res = np.load( fname )
    print 'S_full_res.shape : ', S_full_res.shape


    ##
    ## Set Keras Model
    ##
    image_rows = S_full_res.shape[1]
    image_cols = S_full_res.shape[2]
    input_img = keras.layers.Input( shape=(image_rows, image_cols, 3 ) )
    cnn = make_upsampling_vgg( input_img )
    out = NetVLADLayer(num_clusters = 16)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=out )
    model.summary()
    model.load_weights( 'model.keras/core_model.keras' )


    # Predict for all images
    L = []
    for i in range( S_full_res.shape[0] ):
        IM =  np.expand_dims( S_full_res[i,:,:,:], 0 )
        print IM.shape
        cv2.imshow( 'image', S_full_res[i,:,:,:].astype('uint8') )

        start_time = time.time()
        out, out_amap = model.predict( IM )
        print 'predicted in %4.2fms' %( 1000. * (time.time() - start_time) )
        out_amap_lut = colx.lut( out_amap[0,:,:] )
        cv2.imshow( 'out_amap_lut', out_amap_lut )
        L.append( out[0,:] )

        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    L = np.array( L )
    outfilename = base+'/'+seq+'.txt'
    print 'Save L.shape=%s as file=%s' %( str(L.shape), outfilename )
    np.savetxt( outfilename, L)


if __name__ == '__main__':
    nP = 3
    nN = 1

    #------
    # Load data on RAM
    #------

    PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
    pr = PittsburgRenderer( PTS_BASE )

    # TTM_BASE = '/Bulk_Data/data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
    # pr = TimeMachineRender( TTM_BASE )

    a, _ = pr.step( nP=nP, nN=nN, return_gray=False, resize=None, apply_distortions=False, ENABLE_IMSHOW=False )

    input_img = keras.layers.Input( shape=(480, 640, 3 ) )
    # input_img = keras.layers.Input( shape=(240, 320, 3 ) )
    # cnn = input_img
    # cnn = make_vgg( input_img )
    cnn = make_upsampling_vgg( input_img )

    out = NetVLADLayer(num_clusters = 16)( cnn )
    # code.interact( local=locals() )
    model = keras.models.Model( inputs=input_img, outputs=out )
    model.summary()

    # model.load_weights( 'model.keras/core_model_tryh.keras' )
    # model.load_weights( 'model.keras/core_model_tryh.keras' )
    model.load_weights( 'model.keras/core_model_dataaug.keras' )

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
