""" Testing tf.multiply broadcast capability
    in Patricular X_10x3 s_10x1. Test tf.multiply( X, s)
"""


import numpy as np
import tensorflow as tf

X = np.floor( np.random.rand(10,3)*10 )
s = np.floor( np.random.rand(10,1) * 10 )

tf_X = tf.constant( X )
tf_s = tf.constant( s )

mul_op = tf.multiply( tf_s, tf_X)

sess = tf.Session()
tff_mul = sess.run( mul_op )

print 'X:\n', X
print 's:\n', s
print 'tff_mul:\n', tff_mul
