# A demo script to convert keras model to tensorflow's .pb.
# The goal of this is to load keras model with tensorRT on TX2.
# https://github.com/csvance/keras-tensorrt-jetson
#
# Keras model ---> Tensorflow .pb ---> Nvidia's .uff
#
# This script works fine on my docker image: mpkuse/kusevisionkit:tfgpu-1.12-tensorrt-5.1



import keras
import json
import pprint
import numpy as np
#import cv2
import code

from CustomNets import NetVLADLayer, GhostVLADLayer
from CustomNets import make_from_mobilenet, make_from_vgg16
from predict_utils import open_json_file, change_model_inputshape




LOG_DIR = 'models.keras/May2019/centeredinput-m1to1-240x320x1__mobilenet-conv_pw_7_relu__K16__allpairloss/'
LOG_DIR = 'models.keras/'

##----------------------------------------------------------------------------##
##      LOAD KERAS MODEL
##----------------------------------------------------------------------------##

if False:
    # Load JSON formatted model
    model_json_fname = LOG_DIR+'/model.json'
    print 'Load model_json_fname:', model_json_fname
    json_string = open_json_file( model_json_fname )
    model = keras.models.model_from_json(str(json_string),  custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer': GhostVLADLayer} )
    if False: # printing
        print '======================='
        pprint.pprint( json_string, indent=4 )
        print '======================='
        print 'OLD MODEL: '
        model.summary()

    # Load Weights
    model_fname = LOG_DIR+'/core_model.%d.keras' %(1000)
    print 'Load Weights: ', model_fname
    model.load_weights(  model_fname )
    # Save .h5. The output contains the model defination as well as weights
    # model.save( LOG_DIR+"/modelarch_and_weights.h5" )


if True:
    model_h5_fname = LOG_DIR+'/modelarch_and_weights.h5'
    print 'Load model_h5_fname: ', model_h5_fname
    model = keras.models.load_model(model_h5_fname, custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer': GhostVLADLayer}  )
    model.summary();



# Optionally change_model_inputshape
# TODO



##------------------------------------------------------------------------------##
##  Keras model to tensorflow .pb
##------------------------------------------------------------------------------##
# Code borrowed from : https://github.com/amir-abdi/keras_to_tensorflow
# .h5 to .pb . Freeze
# Use https://github.com/amir-abdi/keras_to_tensorflow
# Also see: https://github.com/csvance/keras-tensorrt-jetson

from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
K.set_learning_phase(0)
sess = K.get_session()

output_model_pbtxt_name = 'output_model.pbtxt'
print 'Write ', output_model_pbtxt_name
tf.train.write_graph(sess.graph.as_graph_def(), LOG_DIR,
                     output_model_pbtxt_name, as_text=True)

output_model_name = 'output_model.pb'
constant_graph = graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    [node.op.name for node in model.outputs])
print 'Write ', output_model_name
graph_io.write_graph(constant_graph, LOG_DIR, output_model_name,
                     as_text=False)

print 'model.outputs=', [node.op.name for node in model.outputs]
print '\t\tcd %s ; convert-to-uff %s ; cd -' %( LOG_DIR, output_model_name )

quit()

FLAGS = {}
FLAGS['output_nodes_prefix'] = 'resulting'
FLAGS['output_meta_ckpt'] = False
FLAGS['save_graph_def'] = True
FLAGS['quantize'] = False
output_model_pbtxt_name = 'output_model.pbtxt'
output_model_name = 'output_model.pb'

orig_output_node_names = [node.op.name for node in model.outputs]
if FLAGS['output_nodes_prefix']:
    num_output = len(orig_output_node_names)
    pred = [None] * num_output
    converted_output_node_names = [None] * num_output

    # Create dummy tf nodes to rename output
    for i in range(num_output):
        converted_output_node_names[i] = '{}{}'.format(
            FLAGS['output_nodes_prefix'], i)
        pred[i] = tf.identity(model.outputs[i],
                              name=converted_output_node_names[i])

else:
    converted_output_node_names = orig_output_node_names
print '**Converted output node names are: %s**' %(str(converted_output_node_names))

sess = K.get_session()
if FLAGS['output_meta_ckpt']:
    saver = tf.train.Saver()
    saver.save(sess, str(output_fld / output_model_stem))

if FLAGS['save_graph_def']:

    tf.train.write_graph(sess.graph.as_graph_def(), LOG_DIR,
                         output_model_pbtxt_name, as_text=True)
    print 'Saved the graph definition in ascii format at %s' %(LOG_DIR+'/'+str(output_model_pbtxt_name))


if FLAGS['quantize']:
    from tensorflow.tools.graph_transforms import TransformGraph
    transforms = ["quantize_weights", "quantize_nodes"]
    transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                           converted_output_node_names,
                                           transforms)
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        transformed_graph_def,
        converted_output_node_names)
else:
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        converted_output_node_names)

graph_io.write_graph(constant_graph, LOG_DIR, output_model_name,
                     as_text=False)
print 'Saved the freezed graph at %s' %( LOG_DIR+'/'+output_model_name )
print 'Finished....OK!'
print 'Now do'
print '\t\tcd %s\n\t\tconvert-to-uff %s' %( LOG_DIR, output_model_name )


##------------------------------------------------------------------------------##
##      Tensorflow .pb to .uff
##------------------------------------------------------------------------------##
# Use nvidia's tensorRT on x86 (download 5.1) on a separate docker. Install by the
# tar method. It installs tensorrt, uff, graphsurgeon.
#       Then use the utility
#       `convert-to-uff output_model.pb` --> outputs output_model.uff
