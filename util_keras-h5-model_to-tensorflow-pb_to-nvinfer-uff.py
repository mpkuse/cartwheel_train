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

    input_img = keras.layers.Input( shape=(240, 320, 3 ) )
    cnn = make_from_mobilenet( input_img, layer_name='conv_pw_5_relu', weights=None, kernel_regularizer=keras.regularizers.l2(0.01) )
    # cnn = make_from_vgg16( input_img, weights=None, layer_name='block5_pool', kernel_regularizer=keras.regularizers.l2(0.01) )

    # base_model = keras.applications.mobilenet_v2.MobileNetV2( weights=None, include_top=False, input_tensor=input_img )
    # cnn = base_model.get_layer( 'block_11_add' ).output

    model = keras.models.Model( inputs=input_img, outputs=cnn )


    # out = NetVLADLayer(num_clusters = 16)( cnn )
    # model = keras.models.Model( inputs=input_img, outputs=out )


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

    cmd = 'convert-to-uff -t -o %s %s | tee %s' %(uff_output_fname, pb_input_fname,    uff_output_fname+'.log')
    print tcol.HEADER, '[bash run] ', cmd, tcol.ENDC

    os.system( cmd )

    print tcol.WARNING, 'If there are warning above like `No conversion function...`, this means that Nvidias UFF doesnt yet have certain function. Most like in this case your model cannot be run with tensorrt.', tcol.ENDC


def graphsurgeon_cleanup( LOG_DIR, input_model_name='output_model.pb', cleaned_model_name='output_model_aftersurgery.pb' ):
    """ Loads the tensorflow frozen_graph and cleans up with nvidia's graphsurgeon
    """
    assert os.path.isfile( LOG_DIR+'/'+input_model_name ), "[graphsurgeon_cleanup]The .pb file="+str(input_model_name)+" does not exist"

    import graphsurgeon as gs
    print tcol.HEADER, '[graphsurgeon_cleanup] graphsurgeon.__version__', gs.__version__, tcol.ENDC

    DG = gs.DynamicGraph()
    print tcol.OKGREEN, '[graphsurgeon_cleanup] READ tensorflow Graph using graphsurgeon.DynamicGraph: ', LOG_DIR+'/'+input_model_name, tcol.ENDC
    DG.read( LOG_DIR+'/'+input_model_name )


    # Remove control variable first


    all_switch = DG.find_nodes_by_op( 'Switch' )
    DG.forward_inputs( all_switch )
    print 'Write (after graphsurgery) : ', LOG_DIR+'/'+cleaned_model_name
    DG.write( LOG_DIR+'/'+cleaned_model_name )


    if os.path.isdir( LOG_DIR+'/graphsurgeon_cleanup' ):
        pass
    else:
        os.mkdir( LOG_DIR+'/graphsurgeon_cleanup')
    DG.write_tensorboard( LOG_DIR+'/graphsurgeon_cleanup' )


    # import code
    # code.interact( local=locals() )


    print tcol.HEADER, '[graphsurgeon_cleanup] END', tcol.ENDC


# def verify_generated_uff_with_tensorrt_uffparser( ufffilename, uffinput, uffinput_dims, uff_output ):
def verify_generated_uff_with_tensorrt_uffparser( ufffilename ):
    """ Loads the UFF file with TensorRT (py). """
    assert os.path.isfile( ufffilename ), "ufffilename="+ ufffilename+ ' doesnt exist'
    import tensorrt as trt

    print tcol.HEADER, '[verify_generated_uff_with_tensorrt_uffparser] TensorRT version=', trt.__version__, tcol.ENDC

    try:
        uffinput = "input_1"
        uffinput_dims = (3,240,320)
        uffoutput = "conv_pw_5_relu/Relu6"
        # uffoutput = "net_vlad_layer_1/l2_normalize_1"

        TRT_LOGGER = trt.Logger( trt.Logger.WARNING)
        with trt.Builder( TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
            print 'ufffilename=', str( ufffilename)
            print 'uffinput=', str( uffinput), '\t', 'uffinput_dims=', str( uffinput_dims)
            print 'uffoutput=', str( uffoutput)
            parser.register_input( uffinput, uffinput_dims  )
            parser.register_output( uffoutput )
            parser.parse( ufffilename, network )
            pass

        print tcol.OKGREEN, '[verify_generated_uff_with_tensorrt_uffparser] Verified.....!', tcol.ENDC
    except:
        print tcol.FAIL, '[verify_generated_uff_with_tensorrt_uffparser] UFF file=', ufffilename, ' with uffinput=', uffinput , ' uffoutput=', uffoutput , ' cannot be parsed.'



if __name__ == '__main__':
    #---
    # Parse Command line
    parser = argparse.ArgumentParser(description='Convert Keras hdf5 models to .uff models for TensorRT.')
    parser.add_argument('--kerasmodel_h5file', '-h5', type=str, help='The input keras modelarch_and_weights full filename')
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
    # model = load_keras_hdf5_model( kerasmodel_h5file, verbose=True ) #this
    model = load_basic_model()


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
    write_kerasmodel_as_tensorflow_pb( new_model, LOG_DIR=LOG_DIR, output_model_name='output_model.pb' )


    #-----
    # Clean up graph with Nvidia's graphsurgeon
    # currently not in use but might come in handly later...maybe
    # graphsurgeon_cleanup( LOG_DIR=LOG_DIR, input_model_name='output_model.pb', cleaned_model_name='output_model_aftersurgery.pb')

    #-----
    # Write UFF
    convert_to_uff( pb_input_fname=LOG_DIR+'/output_model.pb', uff_output_fname=LOG_DIR+'/output_nvinfer.uff' )
    # convert_to_uff( pb_input_fname=LOG_DIR+'/output_model_aftersurgery.pb', uff_output_fname=LOG_DIR+'/output_nvinfer.uff' )


    #-----
    # Try to load UFF with tensorrt
    verify_generated_uff_with_tensorrt_uffparser( ufffilename=LOG_DIR+'/output_nvinfer.uff' )
