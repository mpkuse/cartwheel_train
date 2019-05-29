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

from CustomNets import NetVLADLayer, GhostVLADLayer
from predict_utils import change_model_inputshape


import TerminalColors
tcol = TerminalColors.bcolors()

import argparse

def load_keras_hdf5_model( kerasmodel_h5file, verbose=True  ):
    """ Loads keras model from a HDF5 file """
    assert os.path.isfile( kerasmodel_h5file ), 'The model weights file doesnot exists or there is a permission issue.'+"kerasmodel_file="+kerasmodel_h5file

    model = keras.models.load_model(kerasmodel_h5file, custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer': GhostVLADLayer}  )

    if verbose:
        model.summary();
        print tcol.OKGREEN, 'Successfully Loaded kerasmodel_h5file: ', tcol.ENDC, kerasmodel_h5file

    return model

def write_kerasmodel_as_tensorflow_pb( model, LOG_DIR ):
    """ Takes as input a keras.models.Model() and writes out
        Tensorflow proto-binary.
    """
    print tcol.HEADER,'[write_kerasmodel_as_tensorflow_pb] Start', tcol.ENDC
    from keras import backend as K
    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    K.set_learning_phase(0)
    sess = K.get_session()

    # Write .pbtxt (for viz only)
    output_model_pbtxt_name = 'output_model.pbtxt' #output_model_name+'.pbtxt' #
    print tcol.OKGREEN, 'Write ', output_model_pbtxt_name, tcol.ENDC
    tf.train.write_graph(sess.graph.as_graph_def(), LOG_DIR,
                      output_model_pbtxt_name, as_text=True)

    # Make const
    print 'Make Computation Graph as Constant and Prune unnecessary stuff from it'
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        [node.op.name for node in model.outputs])


    # Write .pb
    output_model_name = 'output_model.pb'
    print tcol.OKGREEN, 'Write ', output_model_name, tcol.ENDC
    print 'model.outputs=', [node.op.name for node in model.outputs]
    graph_io.write_graph(constant_graph, LOG_DIR, output_model_name,
                     as_text=False)
    print tcol.HEADER, '[write_kerasmodel_as_tensorflow_pb] Done', tcol.ENDC


def convert_to_uff( pb_input_fname, uff_output_fname ):
    """ Uses Nvidia's `convert-to-uff` through os.system.
    This will convert the .pb file (generated from call to `write_kerasmodel_as_tensorflow_pb` )
    and write out .uff file.


    usage: convert-to-uff [-h] [-l] [-t] [--write_preprocessed] [-q] [-d]
                      [-o OUTPUT] [-O OUTPUT_NODE] [-I INPUT_NODE]
                      [-p PREPROCESSOR]
                      input_file

    Converts TensorFlow models to Unified Framework Format (UFF).

    positional arguments:
      input_file            path to input model (protobuf file of frozen GraphDef)

    optional arguments:
      -h, --help            show this help message and exit
      -l, --list-nodes      show list of nodes contained in input file
      -t, --text            write a text version of the output in addition to the
                            binary
      --write_preprocessed  write the preprocessed protobuf in addition to the
                            binary
      -q, --quiet           disable log messages
      -d, --debug           Enables debug mode to provide helpful debugging output
      -o OUTPUT, --output OUTPUT
                            name of output uff file
      -O OUTPUT_NODE, --output-node OUTPUT_NODE
                            name of output nodes of the model
      -I INPUT_NODE, --input-node INPUT_NODE
                            name of a node to replace with an input to the model.
                            Must be specified as:
                            "name,new_name,dtype,dim1,dim2,..."
      -p PREPROCESSOR, --preprocessor PREPROCESSOR
                            the preprocessing file to run before handling the
                            graph. This file must define a `preprocess` function
                            that accepts a GraphSurgeon DynamicGraph as it's
                            input. All transformations should happen in place on
                            the graph, as return values are discarded

    """

    assert os.path.isfile( pb_input_fname ), "The .pb file="+str(pb_input_fname)+" does not exist"

    cmd = 'convert-to-uff -t -o %s %s' %(uff_output_fname, pb_input_fname)
    print tcol.HEADER, '[bash run] ', cmd, tcol.ENDC

    os.system( cmd )




if __name__ == '__main__':
    #---
    # Parse Command line
    parser = argparse.ArgumentParser(description='Convert Keras hdf5 models to .uff models for TensorRT.')
    parser.add_argument('--kerasmodel_h5file', '-h5', type=str, help='The input keras modelarch_and_weights full filename')
    args = parser.parse_args()

    # import code
    # code.interact( local=locals() )
    # quit()


    #---
    # Paths, File Init and other initialize
    #kerasmodel_h5file = 'models.keras/June2019/centeredinput-m1to1-240x320x3__mobilenet-conv_pw_6_relu__K16__allpairloss/modelarch_and_weights.700.h5'
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
    model = load_keras_hdf5_model( kerasmodel_h5file, verbose=True )


    #-----
    # Replace Input Layer's Dimensions
    im_rows = 480
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
    write_kerasmodel_as_tensorflow_pb( new_model, LOG_DIR=LOG_DIR )


    #-----
    # Write UFF
    convert_to_uff( pb_input_fname=LOG_DIR+'/output_model.pb', uff_output_fname=LOG_DIR+'/output_nvinfer.uff' )
