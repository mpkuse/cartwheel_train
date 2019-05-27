## A demo of loading the json model

import keras
import json
import pprint
import numpy as np
import cv2
import code

from CustomNets import NetVLADLayer, GhostVLADLayer
from CustomNets import make_from_mobilenet, make_from_vgg16

from predict_utils import open_json_file, change_model_inputshape

def simple_load_demo():
    """
    The way to a)load json-formatted models b)loading weights c) sample prediction
    """
    # LOG_DIR = 'models.keras/Apr2019/K16_gray_training/'
    LOG_DIR = 'models.keras/Apr2019/gray_conv6_K16Ghost1__centeredinput/'


    # Load JSON formatted model
    json_string = open_json_file( LOG_DIR+'/model.json' )
    print '======================='
    pprint.pprint( json_string, indent=4 )
    print '======================='
    model = keras.models.model_from_json(str(json_string),  custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer': GhostVLADLayer} )
    print 'OLD MODEL: '
    model.summary()
    quit()

    # Load Weights from model-file
    model_fname = LOG_DIR+'/core_model.%d.keras' %(100)
    print 'Load model: ', model_fname
    model.load_weights(  model_fname )


    # Replace Input Layer
    new_model = change_model_inputshape( model, new_input_shape=(1,1500,500,1) )


    # Sample Predict
    # test new model on a random input image
    X = np.random.rand(new_input_shape[0], new_input_shape[1], new_input_shape[2], new_input_shape[3] )
    y_pred = new_model.predict(X)
    print('try predict with a random input_img with shape='+str(X.shape)+ str(y_pred) )


def compare_jsonload_with_staticload():
    # Load Image
    # im = cv2.resize( cv2.imread( '/app/lena.jpg', 0 ), (320,240) )
    im = cv2.imread( '/app/lena.jpg', 0 )
    im_rows, im_cols = im.shape[0:2]
    im_centered = (im - 128.)/ 255.
    im_centered = np.expand_dims( np.expand_dims( im_centered, 0 ), -1 )

    LOG_DIR = './models.keras/Apr2019/centeredinput-gray__mobilenet-conv6__K16__allpairloss/'
    # LOG_DIR = './models.keras/Apr2019/default_model/'
    # LOG_DIR = 'models.keras/Apr2019/gray_conv6_K16Ghost1__centeredinput/'

    model_fname = LOG_DIR+'/core_model.%d.keras' %(1000)
    # model_fname = LOG_DIR+'/mobilenet_conv7_allpairloss.keras'

    # Load JSON
    if True:
        json_string = open_json_file( LOG_DIR+'/model.json' )
        model_json = keras.models.model_from_json(str(json_string),  custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer': GhostVLADLayer} )
        print 'Load model into `model_json`: ', model_fname
        model_json.load_weights(  model_fname )
        model_json1 = change_model_inputshape( model_json, new_input_shape=(1,im_rows,im_cols,1), verbose=True )

    # Load Static
    if True:
        image_nrows = im_rows #240
        image_ncols = im_cols #320
        image_nchnl = 1
        netvlad_num_clusters = 16
        input_img = keras.layers.Input( shape=(image_nrows, image_ncols, image_nchnl ) )
        cnn = make_from_mobilenet( input_img, layer_name='conv_pw_7_relu', weights=None, kernel_regularizer=None, trainable=False )
        out = NetVLADLayer(num_clusters = netvlad_num_clusters)( cnn )
        model_static = keras.models.Model( inputs=input_img, outputs=out )
        print 'Load model into `model_json`: ', model_fname
        model_static.load_weights(  model_fname )

    # a = model_json.predict( im_centered )
    # b = model_static.predict( im_centered )


    code.interact( local=locals() )


if __name__ == '__main__':
    # simple_load_demo()
    compare_jsonload_with_staticload()
