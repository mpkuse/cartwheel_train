## A demo of loading the json model

import keras
import json
import pprint
import numpy as np
import cv2
import code

from CustomNets import NetVLADLayer, GhostVLADLayer
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


if __name__ == '__main__':
    simple_load_demo()
