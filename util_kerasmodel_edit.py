#-----------------------------------------------------------------------------#
# Edit the input shape of the model.
#   Often times the models are trained with 320x240x3 in a fully-convolutional
#   network framework. However, at test time the inputs may vary. At test time
#   it is possible to edit the inputs and reallocate, however that takes time.
#   This utility will help to write to disk the editted models so that
#   at test time time is not wasted for reallocation
#-----------------------------------------------------------------------------#

import keras
import os
from CustomNets import NetVLADLayer, GhostVLADLayer
from predict_utils import change_model_inputshape
import argparse

import TerminalColors
tcol = TerminalColors.bcolors()

import tensorflow as tf


def load_keras_hdf5_model( kerasmodel_h5file, verbose=True, inference_only=False  ):
    """ Loads keras model from a HDF5 file """
    assert os.path.isfile( kerasmodel_h5file ), 'The model weights file doesnot exists or there is a permission issue.'+"kerasmodel_file="+kerasmodel_h5file
    if inference_only:
        K.set_learning_phase(0)

    model = keras.models.load_model(kerasmodel_h5file, custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer': GhostVLADLayer}  )

    if verbose:
        print tcol.OKGREEN+'====\n==== Original Input Model\n====', tcol.ENDC
        model.summary();
        print tcol.OKGREEN, 'Successfully Loaded kerasmodel_h5file: ', tcol.ENDC, kerasmodel_h5file
        print tcol.OKGREEN+ '====\n==== END Original Input Model\n====', tcol.ENDC

    return model



if __name__ == '__main__':

    #---
    # Parse command line
    parser = argparse.ArgumentParser(description='Edit input dimensions for a fully convolutional Keras hdf5 model.')
    parser.add_argument('--kerasmodel_h5file', '-h5',required=True, type=str, help='The input keras modelarch_and_weights full filename')
    parser.add_argument('--rows', '-r', type=int, required=True, help='Number of desired input image rows in new model.')
    parser.add_argument('--cols', '-c', type=int, required=True, help='Number of desired input image rows in new model.')
    args = parser.parse_args()


    #---
    # Path and filename
    kerasmodel_h5file = args.kerasmodel_h5file
    nrows = args.rows
    ncols = args.cols
    LOG_DIR = '/'.join( kerasmodel_h5file.split('/')[0:-1] )
    print tcol.HEADER
    print '##------------------------------------------------------------##'
    print '## kerasmodel_h5file = ', kerasmodel_h5file
    print '## LOG_DIR = ', LOG_DIR
    print '## nrows = ', nrows, '\tncols = ', ncols
    print '##------------------------------------------------------------##'
    print tcol.ENDC


    #---
    # Load HDF5 Keras model
    model = load_keras_hdf5_model( kerasmodel_h5file, verbose=True ) #this


    #---
    # Change Shape
    new_input_shape= (  None, nrows, ncols, model.input.shape[3].value )
    new_model = change_model_inputshape( model, new_input_shape=new_input_shape, verbose=True )
    new_model.summary()


    #---
    # Model Save
    new_model_fname = '.'.join( kerasmodel_h5file.split( '.' )[0:-1] )+'.%dx%dx%d.h5' %(new_input_shape[1], new_input_shape[2], new_input_shape[3])
    print '====\n====Save new_model to:', tcol.OKBLUE, new_model_fname, tcol.ENDC, '\n===='
    # import code
    # code.interact( local=locals() )
    new_model.save( new_model_fname )
    print tcol.OKGREEN+'====DONE===='+tcol.ENDC
