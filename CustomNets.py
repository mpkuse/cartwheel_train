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
import code

from imgaug import augmenters as iaa
import imgaug as ia

# Data
from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer


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

    print 'Model file (MB): %4.2f' %(4 * (trainable_count + non_trainable_count) / 1024**2 )
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
    print 'Total floating point operations (GFLOPS) : %4.3f' %( flops.total_float_ops/1000.**3 )
    # return flops.total_float_ops  # Prints the "flops" of the model.



#--------------------------------------------------------------------------------
# Data
#--------------------------------------------------------------------------------
# TODO : removal 
def dataload_( n_tokyoTimeMachine, n_Pitssburg, nP, nN ):
    D = []
    if n_tokyoTimeMachine > 0 :
        TTM_BASE = '/Bulk_Data/data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
        pr = TimeMachineRender( TTM_BASE )
        print 'tokyoTimeMachine:: nP=', nP, '\tnN=', nN
        for s in range(n_tokyoTimeMachine):
            a,_ = pr.step(nP=nP, nN=nN, return_gray=False, resize=(320,240), apply_distortions=False, ENABLE_IMSHOW=False)
            if s%100 == 0:
                print 'get a sample Tokyo_TM #%d of %d\t' %(s, n_tokyoTimeMachine),
                print a.shape
            D.append( a )

    if n_Pitssburg > 0 :
        PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
        pr = PittsburgRenderer( PTS_BASE )
        print 'Pitssburg nP=', nP, '\tnN=', nN
        for s in range(n_Pitssburg):
            a,_ = pr.step(nP=nP, nN=nN, return_gray=False, resize=(240,320), apply_distortions=False, ENABLE_IMSHOW=False)
            if s %100 == 0:
                print 'get a sample Pitssburg #%d of %d\t' %(s, n_Pitssburg),
                print a.shape
            D.append( a )

    return D


def do_augmentation( D ):
    """ D : Nx(n+p+1)xHxWx3. Return N1x(n+p+1)xHxWx3 """

    n_samples = D.shape[0]
    n_images_per_sample = D.shape[1]

    im_rows = D.shape[2]
    im_cols = D.shape[3]
    im_chnl = D.shape[4]

    E = D.reshape( n_samples*n_images_per_sample,  im_rows,im_cols,im_chnl  )


    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Very basic
    if True:
        seq = iaa.Sequential([
            sometimes( iaa.Crop(px=(0, 50)) ), # crop images from each side by 0 to 16px (randomly chosen)
            # iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
            sometimes( iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
                ) )
        ])
        seq_vbasic = seq

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.

    # Typical
    if True:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True) # apply augmenters in random order
        # seq = sometimes( seq )
        seq_typical = seq

    # Heavy
    if True:
        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.2), # horizontally flip 20% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            #iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.OneOf([
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(
                                exponent=(-4, 0),
                                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                second=iaa.ContrastNormalization((0.5, 2.0))
                            )
                        ]),
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
        seq_heavy = seq

    print 'Add data'
    L = [E]
    print 'seq_vbasic'
    L.append( seq_vbasic.augment_images(E) )
    print 'seq_typical'
    L.append( seq_typical.augment_images(E) )
    print 'seq_typical'
    L.append( seq_typical.augment_images(E) )
    print 'seq_heavy'
    L.append( seq_heavy.augment_images(E) )

    G = [ l.reshape(n_samples, n_images_per_sample, im_rows,im_cols,im_chnl) for l in L ]
    G = np.concatenate( G )
    print 'Input.shape ', D.shape, '\tOutput.shape ', G.shape
    return G

    # for j in range(n_times):
    #     images_aug = seq.augment_images(E)
    #     # L.append( images_aug.reshape( n_samples, n_images_per_sample, im_rows,im_cols,im_chnl  ) )
    #     L.append( images_aug )

    # code.interact( local=locals() )
    return L



def do_typical_data_aug( D ):
    """ D : Nx(n+p+1)xHxWx3. Return N1x(n+p+1)xHxWx3 """
    D = np.array( D )
    assert( len(D.shape) == 5 )
    print '[do_typical_data_aug]', 'D.shape=', D.shape

    n_samples = D.shape[0]
    n_images_per_sample = D.shape[1]

    im_rows = D.shape[2]
    im_cols = D.shape[3]
    im_chnl = D.shape[4]

    E = D.reshape( n_samples*n_images_per_sample,  im_rows,im_cols,im_chnl  )


    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes_2 = lambda aug: iaa.Sometimes(0.2, aug)


    seq = iaa.Sequential( [
        #iaa.Fliplr(0.5), # horizontal flips
        #iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    D = seq.augment_images(E)
    D = D.reshape(n_samples, n_images_per_sample, im_rows,im_cols,im_chnl)
    print '[do_typical_data_aug] Done...!', 'D.shape=', D.shape

    return D




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

    def get_config( self ):
        pass
        # import code
        # code.interact( local=locals() )
        base_config = super(NetVLADLayer, self).get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return dict(list(base_config.items()))


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


def make_from_mobilenet( input_img, weights='imagenet',  trainable=True, kernel_regularizer=keras.regularizers.l2(0.0001), layer_name='conv_pw_7_relu' ):
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
    base_model = keras.applications.vgg16.VGG16(weights=weights, include_top=False, input_tensor=input_img)

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
