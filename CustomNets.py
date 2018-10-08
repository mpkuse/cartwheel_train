"""
    Contains my Implementation of NetVLAD and other CNN netsworks using
    keras2.0 with tensorflow1.11.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 7th Oct, 2018
"""


from keras import backend as K
from keras.engine.topology import Layer
import keras
import code
import numpy as np

import cv2



# Writing your own custom layers
class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return [K.dot(x, self.kernel), K.dot(x, self.kernel)]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]


class NetVLADLayer( Layer ):

    def __init__( self, num_clusters, **kwargs ):
        self.num_clusters = num_clusters
        super(NetVLADLayer, self).__init__(**kwargs)

    def build( self, input_shape ):
        self.K = self.num_clusters
        self.D = input_shape[-1]

        self.kernel = self.add_weight( name='kernel',
                                    shape=(1,1,self.D,self.K),
                                    initializer='uniform',
                                    trainable=True )

        self.bias = self.add_weight( name='bias',
                                    shape=(1,1,self.K),
                                    initializer='uniform',
                                    trainable=True )

        self.C = self.add_weight( name='cluster_centers',
                                shape=[1,1,1,self.D,self.K],
                                initializer='uniform',
                                trainable=True)

    def call( self, x ):
        # soft-assignment.
        s = K.conv2d( x, self.kernel, padding='same' ) + self.bias
        a = K.softmax( s )
        self.amap = K.argmax( a, -1 )
        print 'amap.shape', self.amap.shape

        # Dims used hereafter: batch, H, W, desc_coeff, cluster
        a = K.expand_dims( a, -2 )
        # print 'a.shape=',a.shape

        # Core
        v = K.expand_dims(x, -1) + self.C
        # print 'v.shape', v.shape
        v = a * v
        # print 'v.shape', v.shape
        v = K.sum(v, axis=[1, 2])
        # print 'v.shape', v.shape
        v = K.permute_dimensions(v, pattern=[0, 2, 1])
        # print 'v.shape', v.shape
        #v.shape = None x K x D

        # Normalize v (Intra Normalization)
        v = K.l2_normalize( v, axis=-1 )
        v = K.batch_flatten( v )
        v = K.l2_normalize( v, axis=-1 )

        return [v, self.amap]

    def compute_output_shape( self, input_shape ):
        # return (input_shape[0], self.v.shape[-1].value )
        # return [(input_shape[0], self.K*self.D ), (input_shape[0], self.amap.shape[1].value, self.amap.shape[2].value) ]
        return [(input_shape[0], self.K*self.D ), (input_shape[0], input_shape[1], input_shape[2]) ]


def make_vgg( input_img ):
    r_l2=keras.regularizers.l2(0.01)
    r_l1=keras.regularizers.l1(0.01)

    x_64 = keras.layers.Conv2D( 64, (3,3), padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( input_img )
    x_64 = keras.layers.normalization.BatchNormalization()( x_64 )
    x_64 = keras.layers.Conv2D( 64, (3,3), padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( x_64 )
    x_64 = keras.layers.normalization.BatchNormalization()( x_64 )
    x_64 = keras.layers.MaxPooling2D( pool_size=(2,2), padding='same' )( x_64 )


    x_128 = keras.layers.Conv2D( 128, (3,3), padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( x_64 )
    x_128 = keras.layers.normalization.BatchNormalization()( x_128 )
    x_128 = keras.layers.Conv2D( 128, (3,3), padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( x_128 )
    x_128 = keras.layers.normalization.BatchNormalization()( x_128 )
    x_128 = keras.layers.MaxPooling2D( pool_size=(2,2), padding='same' )( x_128 )


    # x_256 = keras.layers.Conv2D( 256, (3,3), padding='same', activation='relu' )( x_128 )
    # x_256 = keras.layers.normalization.BatchNormalization()( x_256 )
    # x_256 = keras.layers.Conv2D( 256, (3,3), padding='same', activation='relu' )( x_256 )
    # x_256 = keras.layers.normalization.BatchNormalization()( x_256 )
    # x_256 = keras.layers.MaxPooling2D( pool_size=(2,2), padding='same' )( x_256 )

    #
    # x_512 = keras.layers.Conv2D( 512, (3,3), padding='same', activation='relu' )( x_256 )
    # # BN
    # x_512 = keras.layers.Conv2D( 512, (3,3), padding='same', activation='relu' )( x_512 )
    # # BN
    # x_512 = keras.layers.MaxPooling2D( pool_size=(2,2), padding='same' )( x_512 )


    x = keras.layers.Conv2DTranspose( 32, (5,5), strides=4, padding='same' )( x_128 )
    # x = x_128

    return x
