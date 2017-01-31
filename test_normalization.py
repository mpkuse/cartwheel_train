""" Test tf.nn.l2_normalize()

    Created : 26th Jan, 2017
    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import code
import argparse

import tensorflow as tf
import tensorflow.contrib.slim as slim



np.random.seed(1)
XX = np.zeros( (16,5,5,8) )
for i in range(16):
    # for j in range(8):
        # XX[i,:,:,j] = np.floor(np.random.randn(5,5)*10) #np.ones((5,5,8)) * i
    XX[i,:,:,:] = np.ones((5,5,8)) * (i+1)
    XX[i,:,:,:] = np.floor(np.random.randn(5,5,8)*10)


tf_x = tf.constant( XX )
op = tf.nn.l2_normalize( tf_x, dim=3 )


sess = tf.Session()
sess.run(tf.global_variables_initializer())

tff_x = sess.run( op )
