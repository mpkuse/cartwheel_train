# As of May end 2019, I moved to HDF5 models, There is some issue with loading jsonfile model.
# Best is not to use json files for model arch. The HDF5 files contain the model architecture
# as well as weights.

# This is a simple example on how to load HDF5 models are do prediction

import keras
import numpy as np
import cv2
import os

from CustomNets import NetVLADLayer, GhostVLADLayer
from predict_utils import change_model_inputshape


import TerminalColors
tcol = TerminalColors.bcolors()

kerasmodel_file = 'models.keras/June2019/centeredinput-m1to1-240x320x3__mobilenet-conv_pw_6_relu__K16__allpairloss/modelarch_and_weights.600.h5'
assert os.path.isfile( kerasmodel_file ), 'The model weights file doesnot exists or there is a permission issue.'+"kerasmodel_file="+kerasmodel_file

#-----
# Load from HDF5
print tcol.OKGREEN, 'Load model: ', kerasmodel_file, tcol.ENDC
model = keras.models.load_model(  kerasmodel_file, custom_objects={'NetVLADLayer': NetVLADLayer, 'GhostVLADLayer':GhostVLADLayer} )
old_input_shape = model._layers[0].input_shape
print 'OLD MODEL: ', 'input_shape=', str(old_input_shape)

#-----
# Plot png file of arch.
model_visual_fname = None
if model_visual_fname is not None:
    model.summary()
    print 'Writing Model Visual to: ', model_visual_fname
    keras.utils.plot_model( model, to_file=model_visual_fname, show_shapes=True )



#-----
# Replace Input Layer's Dimensions (optional)
im_rows = 480
im_cols = 752
im_chnls = 3
new_model = change_model_inputshape( model, new_input_shape=(1,im_rows,im_cols,im_chnls) )
new_input_shape = new_model._layers[0].input_shape
print 'OLD MODEL: ', 'input_shape=', str(old_input_shape)
print 'NEW MODEL: input_shape=', str(new_input_shape)


#-----
# Sample Predict
# test new model on a random input image. Besure to check the input range of the model, for example [-1,1] or [-0.5,0.5] or [0,255] etc.
X = np.random.rand(new_input_shape[0], new_input_shape[1], new_input_shape[2], new_input_shape[3] )

# --- You might want to do any of these normalizations depending on which model files you use.
# i__image = np.expand_dims( cv_image.astype('float32'), 0 )
# i__image = (np.expand_dims( cv_image.astype('float32'), 0 ) - 128.)/255. [-0.5,0.5]
#i__image = (np.expand_dims( cv_image.astype('float32'), 0 ) - 128.)*2.0/255. #[-1,1]

y_pred = new_model.predict(X)
print('try predict with a random input_img with shape='+str(X.shape)+'\n'+ str(y_pred) )
