"""
    Joint training of image descriptors and pixel descriptors.
    Code made with tf1.11 and keras.

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 4th Oct, 2018
"""

# import tensorflow as tf
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
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


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
        print 'a.shape=',a.shape

        # Core
        v = K.expand_dims(x, -1) + self.C
        print 'v.shape', v.shape
        v = a * v
        print 'v.shape', v.shape
        v = K.sum(v, axis=[1, 2])
        print 'v.shape', v.shape
        v = K.permute_dimensions(v, pattern=[0, 2, 1])
        print 'v.shape', v.shape
        #v.shape = None x K x D

        # Normalize v (Intra Normalization)
        v = K.l2_normalize( v, axis=-1 )
        v = K.batch_flatten( v )
        v = K.l2_normalize( v, axis=-1 )

        return v

    def compute_output_shape( self, input_shape ):
        # return (input_shape[0], self.v.shape[-1].value )
        return (input_shape[0], self.K*self.D )


def make_vgg( input_img ):

    x_64 = keras.layers.Conv2D( 64, (3,3), padding='same', activation='relu' )( input_img )
    # BN
    x_64 = keras.layers.Conv2D( 64, (3,3), padding='same', activation='relu' )( x_64 )
    # BN
    x_64 = keras.layers.MaxPooling2D( pool_size=(2,2), padding='same' )( x_64 )


    x_128 = keras.layers.Conv2D( 128, (3,3), padding='same', activation='relu' )( x_64 )
    # BN
    x_128 = keras.layers.Conv2D( 128, (3,3), padding='same', activation='relu' )( x_128 )
    # BN
    x_128 = keras.layers.MaxPooling2D( pool_size=(2,2), padding='same' )( x_128 )


    x_256 = keras.layers.Conv2D( 256, (3,3), padding='same', activation='relu' )( x_128 )
    # BN
    x_256 = keras.layers.Conv2D( 256, (3,3), padding='same', activation='relu' )( x_256 )
    # BN
    x_256 = keras.layers.MaxPooling2D( pool_size=(2,2), padding='same' )( x_256 )


    x_512 = keras.layers.Conv2D( 512, (3,3), padding='same', activation='relu' )( x_256 )
    # BN
    x_512 = keras.layers.Conv2D( 512, (3,3), padding='same', activation='relu' )( x_512 )
    # BN
    x_512 = keras.layers.MaxPooling2D( pool_size=(2,2), padding='same' )( x_512 )


    x = keras.layers.Conv2DTranspose( 32, (5,5), strides=8, padding='same' )( x_512 )

    return x



from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer


if __name__ == '__main__': # Testing renderers
    nP = 2
    nN = 2


    PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
    pr = PittsburgRenderer( PTS_BASE )
    D = []
    for s in range(2000):
        print 'get a sample #', s
        a,_ = pr.step(nP=nP, nN=nN, return_gray=False, resize=None, apply_distortions=False, ENABLE_IMSHOW=False)
        print a.shape
        D.append( a )
        # for i in range( a.shape[0] ):
        #     print i
        #     cv2.imshow( 'win', a[i,:,:,:].astype('uint8'))
        #     cv2.waitKey(0)

    D = np.array( D )    #50 x (1+nP+nN) x 640 x 480 x 3
    # qu = D[:,0:1,:,:,:]  #50 x 1        x 640 x 480 x 3
    # pu = D[:,1:nP+1,:,:,:] #50 x nP       x 640 x 480 x 3
    # nu = D[:,1+nP:1+nP+nN:,:,:,:]  #50 x nN       x 640 x 480 x 3



# if __name__ == '__main__':

    #-----
    # Setting Up core computation
    #-----
    input_img = keras.layers.Input( shape=(20, 20, 3 ) )
    # cnn = input_img
    cnn = make_vgg( input_img )

    out = NetVLADLayer(num_clusters = 4)( cnn )
    # code.interact( local=locals() )
    model = keras.models.Model( inputs=input_img, outputs=out )

    model.summary()
    keras.utils.plot_model( model, to_file='core.png', show_shapes=True )


    #-----
    # 3way siamese / Loss FUnction
    #-----
    #
    # Inputs


    # TODO: This 480x640 can be infered from D
    image_nrows = D.shape[2] #480
    image_ncols = D.shape[3] #640
    image_nchnl = D.shape[4] #3
    im_q = keras.layers.Input( shape=(image_nrows,image_ncols,image_nchnl), name='query_image' )

    im_P = []
    for i in range(nP):
        im_P.append( keras.layers.Input( shape=(image_nrows,image_ncols,image_nchnl), name='positive_%d' %(i) ) )

    im_N = []
    for i in range( nN ):
        im_N.append( keras.layers.Input( shape=(image_nrows,image_ncols,image_nchnl), name='negative_%d' %(i) ) )

    #
    # Netvlad of query
    outq = model( im_q )

    #
    # Dot products

    # <neta_q, neta_P{i}> \forall i
    outP = []
    for i in range( nP ):
        # outP.append( model( im_P[i] ) )
        outP.append( keras.layers.dot( [ outq, model(im_P[i]) ], -1 ) )

    # <neta_q, neta_N{i}> \forall i
    outN = []
    for i in range( nN ):
        # outN.append( model( im_N[i] ) )
        outN.append( keras.layers.dot( [outq, model(im_N[i])], -1 ) )


    #
    # Pairwise loss
    q_dot_P = keras.layers.Concatenate()(outP) #?xnP
    q_dot_N = keras.layers.Concatenate()(outN) #?xnN

    # keras.layers.Permute((2,1))( keras.layers.RepeatVector(3)( q_dot_P ) )
    __a = keras.layers.RepeatVector(nN)( q_dot_P )
    __b = keras.layers.RepeatVector(nP)( q_dot_N )
    minu = keras.layers.Subtract()( [keras.layers.Permute( (2,1) )( __b ), __a ] )

    minu = keras.layers.Lambda( lambda x: x + 0.1)( minu )
    minu = keras.layers.Activation('relu')( minu)
    minu = keras.layers.Lambda( lambda x: K.expand_dims( K.sum( x, axis=[1,2] ), -1 ) )( minu )


    model_tripletloss = keras.models.Model( inputs=[im_q]+im_P+im_N, outputs=minu )
    keras.utils.plot_model( model_tripletloss, to_file='model_tripletloss.png', show_shapes=True )



    #----
    # Iterations
    #----
    # optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)
    model_tripletloss.compile(loss='mean_squared_error', optimizer='sgd')

    opt_inputs = {}
    opt_inputs['query_image'] = D[:,0,:,:,:]
    for i in range(nP):
        opt_inputs['positive_%d' %(i) ] = D[:,1+i,:,:,:]
    for i in range(nN):
        opt_inputs['negative_%d' %(i) ] = D[:,1+nP+i,:,:,:]


    model_tripletloss.fit( opt_inputs, np.zeros( D.shape[0] ),
                        epochs=20, batch_size=5, verbose=2 )
    # X = np.random.random( (2,20,20,6) )
    # u = model.predict( X[0:1,:,:,:] )
    # v = model.predict( X[1:2,:,:,:] )
    # w = model.predict( X )





if __name__ == '__1main__':
    input_img = keras.layers.Input( shape=(256,) )

    out = MyLayer( 15 )( input_img )
    model = keras.models.Model( inputs=input_img, outputs=out )

    model.summary()
    keras.utils.plot_model( model, show_shapes=True )
