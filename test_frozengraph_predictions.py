# Load a .pb (Frozen protobuf) and do inference.
# https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/



import TerminalColors
tcol = TerminalColors.bcolors()

import tensorflow as tf
from tensorflow.python.platform import gfile
from keras import backend as K

import numpy as np
import time

PB_PATH = 'models.keras/June2019/centeredinput-m1to1-240x320x3__mobilenetv2-block_9_add__K16__allpairloss/'
PB_FNAME = PB_PATH+'/'+'modelarch_and_weights.800.480x640x3.pb'


#---
# Load .pb (protobuf file)
print tcol.OKGREEN , 'READ: ', PB_FNAME, tcol.ENDC
f = gfile.FastGFile(PB_FNAME, 'rb')
graph_def = tf.GraphDef()
# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
f.close()

#---
# Setup computation graph
sess = K.get_session()
sess.graph.as_default()
# Import a serialized TensorFlow `GraphDef` protocol buffer
# and place into the current default `Graph`.
tf.import_graph_def(graph_def)


#---
# Print the graph
print tcol.OKGREEN, "=== All Nodes in tf.graph ===",tcol.ENDC
for name in  [n.name for n in tf.get_default_graph().as_graph_def().node]:
    print name
print tcol.OKGREEN, "=== END All Nodes in tf.graph ===", tcol.ENDC
print 'note: The input/output tensors will have the name as opname:0 for example'


#---
# Prediction
print tcol.OKGREEN, "=== sess.run ===",tcol.ENDC
softmax_tensor = sess.graph.get_tensor_by_name('import/net_vlad_layer_1/l2_normalize_1:0')

x_test = np.random.random( (1,480,640,3) )

n_inference = 100
start_t = time.time()
for _ in range(n_inference): #do 10 inferences
    predictions = sess.run(softmax_tensor, {'import/input_1:0': x_test})
end_t = time.time()

print tcol.BOLD, 'x_test.shape=', x_test.shape , '---->' , 'predictions.shape=', predictions.shape, tcol.ENDC
print n_inference, ' inference took: (ms)', 1000.* (end_t - start_t )
