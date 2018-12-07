# Get the trained model from Relja https://www.di.ens.fr/willow/research/netvlad/ (best model)
# and translate into keras

import keras
import code
import time
import numpy as np
import cv2

import scipy.io
import glob
import os.path

# CustomNets
from CustomNets import NetVLADLayer
from CustomNets import make_from_vgg16, make_from_mobilenet

def make_keras_model_relja_netvlad(DATA_DIR, im_rows = 240, im_cols = 320, im_chnls = 3 ):
    # im_rows = 240
    # im_cols = 320
    # im_chnls = 3
    input_img = keras.layers.Input( batch_shape=(1,im_rows, im_cols, im_chnls ) )
    cnn = make_from_vgg16( input_img, weights=None, layer_name='block5_pool' )
    out, out_amap = NetVLADLayer(num_clusters = 64)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=out )

    model.load_weights( DATA_DIR+'/matlab_model.keras' )
    WPCA_M = scipy.io.loadmat( DATA_DIR+'/WPCA_1.mat' )['the_mat'] # 1x1x32768x4096
    WPCA_b = scipy.io.loadmat( DATA_DIR+'/WPCA_2.mat' )['the_mat'] # 4096x1
    WPCA_M = WPCA_M[0,0]          # 32768x4096
    WPCA_b = np.transpose(WPCA_b) #1x4096

    return model, WPCA_M, WPCA_b


DATA_DIR = './relja_matlab_weight.dump/'

# Make a keras model
model, WPCA_M, WPCA_b = make_keras_model_relja_netvlad( DATA_DIR )



U = []
for i, fname_prefix in enumerate( ['a0', 'a1', 'b0', 'b1', 'c0', 'c1', 'd0', 'd1' ] ):
    ## Load Image
    fname = 'sample_images/'+fname_prefix+'.jpg'
    print i, 'Load Image : ', fname
    input_img_bgr = cv2.imread( fname )
    input_img = cv2.cvtColor(input_img_bgr, cv2.COLOR_BGR2RGB)

    ## Normalize Image
    avg_image = [122.6778, 116.6522, 103.9997]
    input_img[:,:,0]= input_img[:,:,0] - avg_image[0]
    input_img[:,:,1]= input_img[:,:,1] - avg_image[1]
    input_img[:,:,2]= input_img[:,:,2] - avg_image[2]

    ## Predict
    u = np.matmul( model.predict( np.expand_dims( input_img, 0) ) , WPCA_M ) + WPCA_b
    u = u[0]
    u = u / np.linalg.norm( u )
    print 'u.shape=', u.shape
    U.append( u )
