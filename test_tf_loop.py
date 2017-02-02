""" Test tf.while_loop
    Adopted from : http://stackoverflow.com/questions/41604686/how-to-use-tf-while-loop-for-variable-length-inputs-in-tensorflow

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 31st Jan, 2017
"""

import tensorflow as tf
import numpy as np


inputs = tf.placeholder(dtype='float32', shape=(None))


time_steps = tf.shape(inputs)[1]
initial_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps)
initial_t = tf.placeholder( dtype='int32')


def should_continue(t, *args):
    return t<time_steps

def iteration(t, outputs_):
    cur = tf.gather_nd(inputs, t)
    cur = tf.slice( inputs, [0, t], [2,1] )
    outputs_ = outputs_.write(t, tf.reduce_sum(cur) )
    return t+1,outputs_

t, outputs = tf.while_loop(should_continue, iteration, [initial_t, initial_outputs])
outputs = outputs.pack()



sess = tf.Session()
sess.run( tf.global_variables_initializer() )
np.random.seed(1)
np_inp = np.floor(np.random.rand( 2,5 )*10)
a_ = sess.run( outputs, feed_dict={inputs: np_inp, initial_t:0} )
