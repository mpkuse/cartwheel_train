""" Testing tf.multiply broadcast capability
    in Parti0cular X_10x3 s_10x1. Test tf.multiply( X, s)
"""


import numpy as np
import tensorflow as tf

X = np.floor( np.random.rand(10,3)*10 )
s = np.floor( np.random.rand(10,1) * 10 )
q = np.floor( np.random.rand(1,3) * 10 )

tf_X = tf.constant( X )
tf_s = tf.constant( s )
tf_q = tf.constant( q )

mul_op = tf.multiply( tf_q, tf_X)



sess = tf.Session()
tff_mul = sess.run( mul_op )

print 'X:\n', X
# print 's:\n', s
print 'q:\n', q
print 'tff_mul:\n', tff_mul
