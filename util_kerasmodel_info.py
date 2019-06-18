#------------------------------------------------------------------------------#
# This script prints out info on the specified model.
#   Can give you:
#   a) Input output dimensions
#   b) Sample Computation times for various input dimensions on your GPU
#   c) FLOPS for the computation
#
#       Author  : Manohar Kuse <mpkuse@connect.ust.hk>
#       Created : 4th June, 2019
#
#------------------------------------------------------------------------------#

import keras
import numpy as np
import os
import tensorflow as tf
from CustomNets import NetVLADLayer, GhostVLADLayer
from predict_utils import change_model_inputshape
from keras import backend as K
import time
import code
import TerminalColors
tcol = TerminalColors.bcolors()

import argparse



def load_keras_hdf5_model( kerasmodel_h5file, verbose=True  ):
    """ Loads keras model from a HDF5 file """
    assert os.path.isfile( kerasmodel_h5file ), 'The model weights file doesnot exists or there is a permission issue.'+"kerasmodel_file="+kerasmodel_h5file
    # K.set_learning_phase(0)

    model = keras.models.load_model(kerasmodel_h5file, custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer': GhostVLADLayer}  )

    if verbose:
        print tcol.OKGREEN+'====\n==== Original Input Model\n====', tcol.ENDC
        model.summary();
        print tcol.OKGREEN, 'Successfully Loaded kerasmodel_h5file: ', tcol.ENDC, kerasmodel_h5file
        print tcol.OKGREEN+ '====\n==== END Original Input Model\n====', tcol.ENDC

    return model

def do_inference_on_random_input( model ):
    input_shape= (  1, int(model.input.shape[1].value), int(model.input.shape[2].value), model.input.shape[3].value )


    start_t = time.time()
    for n in range(100):
        X = np.random.random(input_shape)
        y_pred = model.predict(X)
    end_t = time.time()

    print 'Prediction with random input with shape=%s took %4.2f ms (100 predictions including random generation time) and resulted in output vector of dimensions=%s' %( str(X.shape), 1000. * (end_t-start_t ), str(y_pred.shape) )
    # print('try predict with a random input_img with shape='+str(X.shape)+'\n'+ str(y_pred) )



if __name__ == '__main__':
    #---
    # Parse command line
    parser = argparse.ArgumentParser(description='Print Memory and FLOPS related info on the Keras hdf5 models.')
    parser.add_argument('--kerasmodel_h5file', '-h5', type=str, required=True, help='The input keras modelarch_and_weights full filename')
    args = parser.parse_args()


    #---
    # Path and filename
    kerasmodel_h5file = args.kerasmodel_h5file
    LOG_DIR = '/'.join( kerasmodel_h5file.split('/')[0:-1] )
    print tcol.HEADER
    print '##------------------------------------------------------------##'
    print '## kerasmodel_h5file = ', kerasmodel_h5file
    print '## LOG_DIR = ', LOG_DIR
    print '##------------------------------------------------------------##'
    print tcol.ENDC


    #---
    # Load HDF5 Keras model
    model = load_keras_hdf5_model( kerasmodel_h5file, verbose=True ) #this
    model.summary()



    #---
    # FLOPS
    from CustomNets import print_model_memory_usage, print_flops_report
    print tcol.OKGREEN+ '====\n==== print_model_memory_usage\n====', tcol.ENDC
    print_model_memory_usage( 1, model )
    print_flops_report( model )



    #---
    # Change Model Shape and do flops computation
    print tcol.OKGREEN+ '====\n==== Memory and FLOPS for various Input Shapes\n====', tcol.ENDC
    for m in [0.5, 1.0, 2.0, 4.0]:
        from predict_utils import change_model_inputshape
        new_input_shape= (  None, int(model.input.shape[1].value*m), int(model.input.shape[2].value*m), model.input.shape[3].value )
        new_model = change_model_inputshape( model, new_input_shape=new_input_shape )

        print_model_memory_usage( 1, new_model )
        do_inference_on_random_input( new_model )
