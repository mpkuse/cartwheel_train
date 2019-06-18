#-------------------------------------------------------------------------------#
# Utility to convert Keras model to Tensorflow's .PB (proto-binary) and then to
#       Nvidia libnvinfer's uff format. With UFF one can execute models on
#       TensorRT compatible devices like TX2.
#
#   Author : Manohar Kuse <mpkuse@connect.ust.hk>
#   Created: 29th May, 2019
#   Site   : https://kusemanohar.wordpress.com/2019/05/25/hands-on-tensorrt-on-nvidiatx2/
#-------------------------------------------------------------------------------#

import keras
import numpy as np
import os
import tensorflow as tf
from CustomNets import NetVLADLayer, GhostVLADLayer
from predict_utils import change_model_inputshape
from keras import backend as K

import TerminalColors
tcol = TerminalColors.bcolors()

import argparse

def load_keras_hdf5_model( kerasmodel_h5file, verbose=True  ):
    """ Loads keras model from a HDF5 file """
    assert os.path.isfile( kerasmodel_h5file ), 'The model weights file doesnot exists or there is a permission issue.'+"kerasmodel_file="+kerasmodel_h5file
    K.set_learning_phase(0)

    model = keras.models.load_model(kerasmodel_h5file, custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer': GhostVLADLayer}  )

    if verbose:
        model.summary();
        print tcol.OKGREEN, 'Successfully Loaded kerasmodel_h5file: ', tcol.ENDC, kerasmodel_h5file

    return model


def load_basic_model( ):
    K.set_learning_phase(0)
    from CustomNets import make_from_mobilenet, make_from_vgg16
    from CustomNets import NetVLADLayer, GhostVLADLayer

    # Please choose only one of these.
    if False: # VGG
        input_img = keras.layers.Input( shape=(240, 320, 3 ) )
        cnn = make_from_vgg16( input_img, weights=None, layer_name='block5_pool', kernel_regularizer=keras.regularizers.l2(0.01) )
        model = keras.models.Model( inputs=input_img, outputs=cnn )

    if True: #mobilenet
        input_img = keras.layers.Input( shape=(240, 320, 3 ) )
        cnn = make_from_mobilenet( input_img, layer_name='conv_pw_5_relu', weights=None, kernel_regularizer=keras.regularizers.l2(0.01) )
        model = keras.models.Model( inputs=input_img, outputs=cnn )

    if False: #mobilenet+netvlad
        input_img = keras.layers.Input( shape=(240, 320, 3 ) )
        cnn = make_from_mobilenet( input_img, layer_name='conv_pw_5_relu', weights=None, kernel_regularizer=keras.regularizers.l2(0.01) )
        # cnn = make_from_vgg16( input_img, weights=None, layer_name='block5_pool', kernel_regularizer=keras.regularizers.l2(0.01) )
        out = NetVLADLayer(num_clusters = 16)( cnn )
        model = keras.models.Model( inputs=input_img, outputs=out )

    if False: #netvlad only
        input_img = keras.layers.Input( shape=(60, 80, 256 ) )
        out = NetVLADLayer(num_clusters = 16)( input_img )
        model = keras.models.Model( inputs=input_img, outputs=out )




    model.summary()
    return model

def write_kerasmodel_as_tensorflow_pb( model, LOG_DIR, output_model_name='output_model.pb' ):
    """ Takes as input a keras.models.Model() and writes out
        Tensorflow proto-binary.
    """
    print tcol.HEADER,'[write_kerasmodel_as_tensorflow_pb] Start', tcol.ENDC

    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    K.set_learning_phase(0)
    sess = K.get_session()



    # Make const
    print 'Make Computation Graph as Constant and Prune unnecessary stuff from it'
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        [node.op.name for node in model.outputs])
    constant_graph = tf.graph_util.remove_training_nodes(constant_graph)


    #--- convert Switch --> Identity
    # I am doing this because TensorRT cannot process Switch operations.
    # # https://github.com/tensorflow/tensorflow/issues/8404#issuecomment-297469468
    # for node in constant_graph.node:
    #     if node.op == "Switch":
    #         node.op = "Identity"
    #         del node.input[1]
    # # END

    # Write .pb
    # output_model_name = 'output_model.pb'
    print tcol.OKGREEN, 'Write ', output_model_name, tcol.ENDC
    print 'model.outputs=', [node.op.name for node in model.outputs]
    graph_io.write_graph(constant_graph, LOG_DIR, output_model_name,
                     as_text=False)
    print tcol.HEADER, '[write_kerasmodel_as_tensorflow_pb] Done', tcol.ENDC


    # Write .pbtxt (for viz only)
    output_model_pbtxt_name = output_model_name+'.pbtxt' #'output_model.pbtxt'
    print tcol.OKGREEN, 'Write ', output_model_pbtxt_name, tcol.ENDC
    tf.train.write_graph(constant_graph, LOG_DIR,
                      output_model_pbtxt_name, as_text=True)

    # Write model.summary to file (to get info on input and output shapes)
    output_modelsummary_fname = LOG_DIR+'/'+output_model_name + '.modelsummary.log'
    print tcol.OKGREEN, 'Write ', output_modelsummary_fname, tcol.ENDC
    with open(output_modelsummary_fname,'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

if __name__ == '__main__':
    #---
    # Parse Command line
    parser = argparse.ArgumentParser(description='Convert Keras hdf5 models to .uff models for TensorRT.')
    parser.add_argument('--kerasmodel_h5file', '-h5', required=True, type=str, help='The input keras modelarch_and_weights full filename')
    args = parser.parse_args()



    #---
    # Paths, File Init and other initialize
    # kerasmodel_h5file = 'models.keras/June2019/centeredinput-m1to1-240x320x3__mobilenet-conv_pw_6_relu__K16__allpairloss/modelarch_and_weights.700.h5'
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
    # model = load_basic_model()
    # quit()

    #-----
    # Replace Input Layer's Dimensions
    im_rows = None#480
    im_cols = 752
    im_chnls = 3
    if im_rows == None or im_cols == None or im_chnls == None:
        print tcol.WARNING, 'NOT doing `change_model_inputshape`', tcol.ENDC
        new_model = model
    else:
        # change_model_inputshape uses model_from_json internally, I feel a bit uncomfortable about this.
        new_model = change_model_inputshape( model, new_input_shape=(1,im_rows,im_cols,im_chnls), verbose=True )
    print 'OLD MODEL: ', 'input_shape=', str(model.inputs)
    print 'NEW MODEL: input_shape=', str(new_model.inputs)


    #-----
    # Write Tensorflow (atleast 1.12) proto-binary (.pb)
    # write_kerasmodel_as_tensorflow_pb( new_model, LOG_DIR=LOG_DIR, output_model_name='output_model.pb' )
    out_pb_fname = '.'.join(    (kerasmodel_h5file.split('/')[-1]).split('.')[:-1]    )+'.pb'
    write_kerasmodel_as_tensorflow_pb( new_model, LOG_DIR=LOG_DIR, output_model_name=out_pb_fname )
