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
# from CustomNets import make_vgg, make_upsampling_vgg, make_from_vgg19, make_from_vgg19_multiconvup
from CustomNets import make_from_mobilenet



# Data
from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer

# Custom Utils
from ColorLUT import ColorLUT
colx = ColorLUT()

from imgaug import augmenters as iaa
import imgaug as ia


# Test on Tokyo TM
if __name__ == '__main__':

    #---
    # Set network
    #---
    input_img = keras.layers.Input( shape=(480, 640, 3 ) )


    cnn = make_from_mobilenet( input_img )
    out, out_amap = NetVLADLayer(num_clusters = 16)( cnn )


    # cnn = make_from_vgg19_multiconvup( input_img, trainable=True )
    # out = NetVLADLayer(num_clusters = 16)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=[out, out_amap] )
    model.summary()


    # model.load_weights( 'models.keras/model_with_dataaug_batchnorm/core_model_vgg19pretained_fixedvgglayers.keras' )
    # model.load_weights( 'models.keras/model_learn_with_regul_multi_samplefit/core_model.keras' )
    # model.load_weights( 'models.keras/mobilenet_conv7_allpairloss/core_model.keras' )
    model.load_weights( 'models.keras/mobilenet_conv7_tripletloss2/core_model.keras' )



    #---
    # Dat Aug
    #---
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        sometimes( iaa.Crop(px=(0, 50)) ), # crop images from each side by 0 to 16px (randomly chosen)
        # iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        # sometimes( iaa.Affine(
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     rotate=(-25, 25),
        #     shear=(-8, 8)
        #     ) )
    ])

    WR_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
    wr = PittsburgRenderer( WR_BASE )

    # TTM_BASE = '/Bulk_Data/data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
    # wr = TimeMachineRender( TTM_BASE )

    for i in range(20):
        nP = 5
        nN = 5
        a,b = wr.step(nP=nP, nN=nN,return_gray=False, resize=None, apply_distortions=False, ENABLE_IMSHOW=False)

        a = seq.augment_images( a )
        print a.shape
        print b.shape

        out, out_amap = model.predict( a[:,:,:,:] )
        M = np.matmul( out[0:1,:], np.transpose( out[1:,:] ) )
        print '<q,Pi>', M[0,:nP]
        print '<q,Ni>', M[0,nP:]

        # code.interact( local=locals() )

        for k in range(1,1+nP+nN):
            a[k,:,:,:] = cv2.putText(a[k,:,:,:].astype('uint8'),str(np.round(M[0,k-1],2)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2 )
        # for k in range(nP+1,nN+nP+1):
            # a[k,:,:,:] = cv2.putText(a[k,:,:,:].astype('uint8'),str(np.round(M[0,k],2)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2 )

        CAT_sims = np.concatenate( a[:1+nP,:,:,:], axis=1).astype('uint8')
        CAT_diffs = np.concatenate( a[1+nP:], axis=1).astype('uint8')
        cv2.imshow( 'sims_im', cv2.resize( CAT_sims , (0,0), fx=.5, fy=.5 ) )
        cv2.moveWindow( 'sims_im', 20,20 )
        cv2.imshow( 'diff_im', cv2.resize( CAT_diffs , (0,0), fx=.5, fy=.5 ) )
        cv2.moveWindow( 'diff_im', 20,350 )
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

if __name__ == '__1main__': # Use __seq__


    ##
    ## Load Data
    ##
    base = '/Bulk_Data/__seq_nap_out__/'
    seq = 'base-2'
    # seq = 'tpt-park'
    # seq = 'coffee-shop'
    # seq = 'lsk-1'

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
    # cnn = make_upsampling_vgg( input_img )
    # cnn = make_from_vgg19( input_img, trainable=False )
    # cnn = make_from_vgg19_multiconvup( input_img, trainable=True )
    cnn = make_from_mobilenet( input_img )

    out, out_amap = NetVLADLayer(num_clusters = 16)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=[out,out_amap]  )
    model.summary()

    # model.load_weights( 'models.keras/mobilenet_test_conv7/core_model.keras' )
    model.load_weights( 'models.keras/mobilenet_conv7_allpairloss/core_model.keras' )



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



# Not in use. TODO: Removal
if __name__ == '__1main__': # Use pits burg or timemachine
    nP = 3
    nN = 3

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
    # cnn = make_upsampling_vgg( input_img )
    # cnn = make_from_vgg19( input_img, trainable=False )
    cnn = make_from_vgg19_multiconvup( input_img, trainable=True )



    out = NetVLADLayer(num_clusters = 16)( cnn )
    # code.interact( local=locals() )
    model = keras.models.Model( inputs=input_img, outputs=out )
    model.summary()


    # model.load_weights( 'model.keras/core_model_dataaug.keras' )
    # model.load_weights( 'model.keras/core_model_vgg19pretained.keras' )
    # model.load_weights( 'model.keras/core_model_vgg19pretained_fixedvgglayers.keras' )
    model.load_weights( 'models.keras/model_with_dataaug_batchnorm/core_model_vgg19pretained_fixedvgglayers.keras' )


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
