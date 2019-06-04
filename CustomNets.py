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

# import cv2
import code

# from imgaug import augmenters as iaa
# import imgaug as ia

# Data
# from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
# from WalksRenderer import WalksRenderer
# from PittsburgRenderer import PittsburgRenderer


#-------------------------------------------------------------------------------
# Utilities
#-------------------------------------------------------------------------------
# Forward pass memory requirement
def print_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers: #loop on layers
        # print '---\n', l
        # print 'out_shapes: ', str( l.output_shape ),
        # print 'isList: ', type(l.output_shape) == type(list()),
        # print 'isTuple: ', type(l.output_shape) == type(tuple())


        all_output_shapes = l.output_shape
        if type(all_output_shapes) != type(list()):
            all_output_shapes = list( [all_output_shapes] )

        for n_out in all_output_shapes:
            single_layer_mem = 1
            for s in n_out: #loop on outputs shape
                if s is None:
                    continue
                single_layer_mem *= s
            # print 'single_layer_mem', single_layer_mem
            shapes_mem_count+= single_layer_mem


        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    print 'Model Inputs: ', str(model.inputs), '\nModel Outputs: ', str(model.outputs)
    print 'Model file (MB): %4.4f' %(4. * (trainable_count + non_trainable_count) / 1024**2 )
    print '#Trainable Params: ', trainable_count
    print 'Layers(batch_size)=%d (MB): %4.2f' %(batch_size, 4.0*batch_size*shapes_mem_count/1024**2 )

    total_memory = 4.0*(batch_size*shapes_mem_count + trainable_count + non_trainable_count) # 4 is multiplied because all the memoery is of data-type float32 (4 bytes)
    print 'Total Memory(MB): %4.2f' %( total_memory/1024**2 )
    # gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    # return gbytes


def print_flops_report(model):
    # Batch need to be specified for flops number to be accurate.
    import tensorflow as tf
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
    print 'Total floating point operations (FLOPS) : ', flops.total_float_ops
    print 'Total floating point operations (GFLOPS) : %4.3f' %( flops.total_float_ops/1000.**2 )
    return flops.total_float_ops  # Prints the "flops" of the model.



#---------------------------------------------------------------------------------
# My Layers
#   NetVLADLayer
#---------------------------------------------------------------------------------

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

    # Experimentation for TensorRT
    # def call( self, x ):
    #     print 'input x.shape=', x.shape
    #     # soft-assignment.
    #     s = K.conv2d( x, self.kernel, padding='same' ) + self.bias
    #     print 's.shape=', s.shape
    #     a = K.softmax( s )
    #     print 'a.shape=',a.shape
    #
    #     self.amap = K.argmax( a, -1 ) #<----- currently not needed for output. if need be uncomment this and will also have to change compute_output_shape
    #     print 'amap.shape', self.amap.shape
    #
    #     # import code
    #     # code.interact( local=locals() )
    #     # Dims used hereafter: batch, H, W, desc_coeff, cluster
    #     print 'a.shape (before)=', a.shape
    #     # a = K.expand_dims( a, -2 ) #original code
    #     # a = K.reshape( a, [ K.shape(a)[0], K.shape(a)[1], K.shape(a)[2], 1, K.shape(a)[3]] ) # I think only for unknown shapes should use K.shape(a)[0] etc
    #     a = K.reshape( a, [ K.shape(a)[0], a.shape[1].value, a.shape[2].value, 1, a.shape[3].value ] )
    #     print 'a.shape=',a.shape
    #
    #
    #     # Core
    #     print 'x.shape', x.shape
    #     # v = K.expand_dims(x, -1) + self.C #original code
    #     v_tmp = K.reshape( x, [ K.shape(x)[0],  x.shape[1].value, x.shape[2].value, x.shape[3].value, 1 ] )
    #     print 'v_tmp.shape', v_tmp.shape, '\tself.C.shape', self.C.shape
    #     v = v_tmp + self.C
    #     print 'v.shape', v.shape
    #     return v
    #     v = a * v
    #     # print 'v.shape', v.shape
    #     v = K.sum(v, axis=[1, 2])
    #     # print 'v.shape', v.shape
    #     v = K.permute_dimensions(v, pattern=[0, 2, 1])
    #     print 'v.shape', v.shape
    #     #v.shape = None x K x D
    #
    #     # Normalize v (Intra Normalization)
    #     v = K.l2_normalize( v, axis=-1 )
    #     v = K.batch_flatten( v )
    #     v = K.l2_normalize( v, axis=-1 )
    #
    #     # return [v, self.amap]
    #     print 'v.shape (final)', v.shape
    #     return v
    #
    # def compute_output_shape( self, input_shape ):
    #
    #     # return [(input_shape[0], self.K*self.D ), (input_shape[0], input_shape[1], input_shape[2]) ]
    #     # return (input_shape[0], self.K*self.D )
    #
    #     # return (input_shape[0], input_shape[1], input_shape[2], 1, self.K) #s
    #     return (input_shape[0], input_shape[1], input_shape[2], self.D, self.K) #s


    # Old code - working fine
    def call( self, x ):
        # print 'input x.shape=', x.shape
        # soft-assignment.
        s = K.conv2d( x, self.kernel, padding='same' ) + self.bias
        a = K.softmax( s )
        self.amap = K.argmax( a, -1 )
        # print 'amap.shape', self.amap.shape

        # import code
        # code.interact( local=locals() )
        # Dims used hereafter: batch, H, W, desc_coeff, cluster
        # print 'a.shape (before)=', a.shape
        a = K.expand_dims( a, -2 ) #original code
        # a = K.reshape( a, [ K.shape(a)[0], K.shape(a)[1], K.shape(a)[2], 1, K.shape(a)[3]] )
        # print 'a.shape=',a.shape

        # Core
        # print 'x.shape', x.shape
        v = K.expand_dims(x, -1) + self.C #original code
        # v = K.reshape( x, [ K.shape(x)[0],  K.shape(x)[1], K.shape(x)[2], K.shape(x)[3], 1 ] ) + self.C
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

        # return [v, self.amap]
        return v

    def compute_output_shape( self, input_shape ):

        # return [(input_shape[0], self.K*self.D ), (input_shape[0], input_shape[1], input_shape[2]) ]
        return (input_shape[0], self.K*self.D )

    def get_config( self ):
        pass
        # base_config = super(NetVLADLayer, self).get_config()
        # return dict(list(base_config.items()))

        # As suggested by: https://github.com/keras-team/keras/issues/4871#issuecomment-269731817
        config = {'num_clusters': self.num_clusters}
        base_config = super(NetVLADLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GhostVLADLayer( Layer ):

    def __init__( self, num_clusters, num_ghost_clusters, **kwargs ):
        self.num_clusters = num_clusters
        self.num_ghost_clusters = num_ghost_clusters
        super(GhostVLADLayer, self).__init__(**kwargs)

    def build( self, input_shape ):
        # self.K = self.num_clusters
        self.K = self.num_clusters + self.num_ghost_clusters
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
        # print 'amap.shape', self.amap.shape

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
        v = v[:,0:self.num_clusters,:]
        # print 'after ghosting v.shape', v.shape
        v = K.l2_normalize( v, axis=-1 )
        v = K.batch_flatten( v )
        v = K.l2_normalize( v, axis=-1 )


        # return [v, self.amap]
        return v

    def compute_output_shape( self, input_shape ):

        # return [(input_shape[0], self.num_clusters*self.D ), (input_shape[0], input_shape[1], input_shape[2]) ]
        return (input_shape[0], self.num_clusters*self.D )

    def get_config( self ):
        pass


        # As suggested by: https://github.com/keras-team/keras/issues/4871#issuecomment-269731817
        config = {'num_clusters': self.num_clusters, 'num_ghost_clusters': self.num_ghost_clusters}
        base_config = super(GhostVLADLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#--------------------------------------------------------------------------------
# Base CNNs
#--------------------------------------------------------------------------------

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


def make_upsampling_vgg( input_img  ):
    r_l2=keras.regularizers.l2(0.01)
    r_l1=keras.regularizers.l1(0.01)

    x_64 = keras.layers.Conv2D( 64, (3,3), padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( input_img )
    x_64 = keras.layers.normalization.BatchNormalization()( x_64 )
    x_64 = keras.layers.Conv2D( 64, (3,3), strides=2, padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( x_64 )
    x_64 = keras.layers.normalization.BatchNormalization()( x_64 )

    x_128 = keras.layers.Conv2D( 128, (3,3), padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( x_64 )
    x_128 = keras.layers.normalization.BatchNormalization()( x_128 )
    x_128 = keras.layers.Conv2D( 128, (3,3), strides=2, padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( x_128 )
    x_128 = keras.layers.normalization.BatchNormalization()( x_128 )

    x_256 = keras.layers.Conv2D( 128, (3,3), padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( x_128 )
    x_256 = keras.layers.normalization.BatchNormalization()( x_256 )
    x_256 = keras.layers.Conv2D( 128, (3,3), strides=2, padding='same', activation='relu', kernel_regularizer=r_l2, activity_regularizer=r_l1 )( x_256 )
    x_256 = keras.layers.normalization.BatchNormalization()( x_256 )

    z = keras.layers.Conv2DTranspose( 32, (11,11), strides=8, padding='same' )( x_256 )
    x = keras.layers.Conv2DTranspose( 32, (9,9), strides=4, padding='same' )( x_128 )
    y = keras.layers.Conv2DTranspose( 32, (7,7), strides=2, padding='same' )( x_64 )

    out = keras.layers.Add()( [x,y,z] )
    return out



def make_from_vgg19_multiconvup( input_img, trainable=True ):
    base_model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_tensor=input_img)

    for l in base_model.layers:
        l.trainable = trainable
        #TODO : add kernel regularizers and activity_regularizer to conv layers

    base_model_out = base_model.get_layer('block2_pool').output

    up_conv_out = keras.layers.Conv2DTranspose( 32, (9,9), strides=2, padding='same', activation='relu' )( base_model_out )
    up_conv_out = keras.layers.normalization.BatchNormalization()( up_conv_out )

    up_conv_out = keras.layers.Conv2DTranspose( 32, (9,9), strides=2, padding='same', activation='relu' )( up_conv_out )
    up_conv_out = keras.layers.normalization.BatchNormalization()( up_conv_out )

    return up_conv_out


def make_from_mobilenet( input_img, weights='imagenet',  trainable=True, kernel_regularizer=keras.regularizers.l2(0.001), layer_name='conv_pw_7_relu' ):
    # input_img = keras.layers.BatchNormalization()(input_img)

    base_model = keras.applications.mobilenet.MobileNet( weights=weights, include_top=False, input_tensor=input_img )

    for l in base_model.layers:
        l.trainable = trainable

    # Add Regularizers
    if kernel_regularizer is not None:
        for layer in base_model.layers:
            if 'kernel_regularizer' in dir( layer ):
                # layer.kernel_regularizer = keras.regularizers.l2(0.001)
                layer.kernel_regularizer = kernel_regularizer


    # Pull out a layer from original network
    base_model_out = base_model.get_layer( layer_name ).output # can also try conv_pw_7_relu etc.

    return base_model_out


def make_from_mobilenetv2( input_img, weights='imagenet',  trainable=True, kernel_regularizer=keras.regularizers.l2(0.001), layer_name='block_9_add' ):
    base_model = keras.applications.mobilenet_v2.MobileNetV2( weights=weights, include_top=False, input_tensor=input_img )

    for l in base_model.layers:
        l.trainable = trainable

    # Add Regularizers
    if kernel_regularizer is not None:
        for layer in base_model.layers:
            if 'kernel_regularizer' in dir( layer ):
                # layer.kernel_regularizer = keras.regularizers.l2(0.001)
                layer.kernel_regularizer = kernel_regularizer

    # Pull out a layer from original network
    base_model_out = base_model.get_layer( layer_name ).output # can also try conv_pw_7_relu etc.

    return base_model_out


def make_from_vgg19( input_img, weights='imagenet', trainable=True, layer_name='block2_pool' ):
    base_model = keras.applications.vgg19.VGG19(weights=weights, include_top=False, input_tensor=input_img)

    for l in base_model.layers:
        l.trainable = trainable

    base_model_out = base_model.get_layer(layer_name).output
    return base_model_out

    # Removal. TODO: Not more in use.
    # z = keras.layers.Conv2DTranspose( 32, (9,9), strides=4, padding='same' )( base_model_out )
    # return z




def make_from_vgg16( input_img, weights='imagenet', trainable=True, kernel_regularizer=keras.regularizers.l2(0.0001), layer_name='block2_pool' ):
    base_model = keras.applications.vgg16.VGG16(include_top=False, weights=weights, input_tensor=input_img)

    for l in base_model.layers:
        l.trainable = trainable

    # Add Regularizers
    if kernel_regularizer is not None:
        for layer in base_model.layers:
            if 'kernel_regularizer' in dir( layer ):
                # layer.kernel_regularizer = keras.regularizers.l2(0.001)
                layer.kernel_regularizer = kernel_regularizer



    base_model_out = base_model.get_layer(layer_name).output
    return base_model_out
