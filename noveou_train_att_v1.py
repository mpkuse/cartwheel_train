""" Training with Attention RNN

    - one-input image, outputs multiple attention maps.

    Code based on:
        https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 22nd Oct, 2018
"""


import keras
from keras.layers import Input, Multiply
from keras.layers import LSTM, ConvLSTM2D
# from keras.layers import LSTMCell


from CustomNets import make_from_mobilenet


import code
import numpy as np
import cv2


image_nrows = 240
image_ncols = 320
image_nchnl = 3

input_img = Input( shape=(50,), name='feature_cube' )
att = Input( shape=(50,), name='attention' )

u = Multiply()( [input_img, att] )

from recurrentshop import LSTMCell
lstm_output, state1_t, state2_t = LSTMCell(10)( u, state1_tm1, state_tm1 )




model = keras.models.Model( inputs=[input_img,att], outputs=u )


model.summary()
keras.utils.plot_model( model, to_file='./model_att.png', show_shapes=True )
