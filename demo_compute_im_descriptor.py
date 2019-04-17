## A demo script which inputs an image and produces its whole image
## descriptor.
## Here is what this demo dones:
## a) Create Model (from json files)
## b) Load learned weights
## c) Display Model Info
## d) Load Image and compute its descriptor

import keras
import code
import time
import numpy as np

# CustomNets
from CustomNets import NetVLADLayer
from CustomNets import make_from_vgg16, make_from_mobilenet
from CustomNets import print_model_memory_usage, print_flops_report

base_path = './models.keras/mobilenet_conv7_allpairloss/'

def do_demo():
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    # config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

    im_rows = 240
    im_cols = 320
    im_chnls = 3

    # im_rows = 480
    # im_cols = 640
    # im_chnls = 3

    ##------ Create Model (from json file)
    # [[[[[Option-A]]]]] (load from json) this doesn't work. seem like i need to implement get_config in my custom layer which I am not able to do correctly, TODO fix in the future
    # json_fname = base_path+'/model.json'
    # print 'Read model json: ', json_fname
    # with open(json_fname, 'r') as myfile:
        # model_json_string=myfile.read()
    # model = keras.models.model_from_json( model_json_string, custom_objects={'NetVLADLayer': NetVLADLayer} )

    # [[[[[Option-B]]]]] (from h5 files). Apparently h5 stores the model config and the weights.
    # h5_fname = base_path+'/model.h5'
    # model = keras.models.load_model( h5_fname, custom_objects={'NetVLADLayer': NetVLADLayer} )

    # [[[[[Option-C]]]]] : Construct manually
    # @ Input
    # input_img = keras.layers.Input( shape=(480, 640, 3 ) )
    # input_img = keras.layers.Input( shape=(240, 320, 3 ) )
    # input_img = keras.layers.Input( batch_shape=(1,480, 640, 3 ) )
    input_img = keras.layers.Input( batch_shape=(1,im_rows, im_cols, im_chnls ) )

    # @ CNN
    # cnn = make_from_vgg16( input_img, weights=None, layer_name='block5_pool' )
    cnn = make_from_mobilenet( input_img, weights=None, layer_name='conv_pw_7_relu' )

    # @ Downsample (Optional)
    if True: #Downsample last layer (Reduce nChannels of the output.)
        cnn_dwn = keras.layers.Conv2D( 256, (1,1), padding='same', activation='relu' )( cnn )
        cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
        cnn_dwn = keras.layers.Conv2D( 32, (1,1), padding='same', activation='relu' )( cnn_dwn )
        cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
        cnn = cnn_dwn

    # @ NetVLADLayer
    out, out_amap = NetVLADLayer(num_clusters = 16)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=out )


    ##---------- Print Model Info, Memory Usage, FLOPS
    if True: # Set this to `True` to display FLOPs, memory etc .
        model.summary()
        keras.utils.plot_model( model, to_file='./demo_.png', show_shapes=True )

        print_flops_report( model )
        print_model_memory_usage( 1, model )
        print 'input_shape=%s\toutput_shape=%s' %( model.input_shape, model.output_shape )

    ##------------ Load Weights



    ##------------model.predict
    # Load your image here.
    tmp_zer = np.random.rand( 1,im_rows,im_cols,im_chnls  ).astype('float32')
    start = time.time()
    tmp_zer_out = model.predict( tmp_zer )
    print 'Exec in %4.2fms' %( 1000. * (time.time() - start ) )

    tmp_zer = np.random.rand( 1,im_rows,im_cols,im_chnls  ).astype('float32')
    start = time.time()
    tmp_zer_out = model.predict( tmp_zer )
    print 'Exec in %4.2fms' %( 1000. * (time.time() - start ) )


if __name__ == '__main__':
    do_demo()
