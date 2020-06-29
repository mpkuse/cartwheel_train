""" NetVLAD training. Using features from deeper VGG layers.

    - Input -- BN -- MobileNet -- NetVLAD
    - Deeper features (instead of currently using very shallow ones)
    - Verify my pairwise_loss also implement triplet loss
    - Keras sequence use

    All the configurable things are in this file itself.

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 15th Oct, 2018
"""


from keras import backend as K
from keras.engine.topology import Layer
import keras

import code
import numpy as np
import cv2
import pickle
import json


# CustomNets
from CustomNets import NetVLADLayer, GhostVLADLayer
from CustomNets import make_from_mobilenet, make_from_vgg16, make_from_mobilenetv2
from CustomDataProc import dataload_, do_typical_data_aug

# CustomLoses
from CustomLosses import triplet_loss2_maker, allpair_hinge_loss_maker, allpair_count_goodfit_maker, positive_set_deviation_maker, allpair_hinge_loss_with_positive_set_deviation_maker

from InteractiveLogger import InteractiveLogger
import TerminalColors
tcolor = TerminalColors.bcolors()

# Data
#from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
#from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer



class PitsSequence(keras.utils.Sequence):
    """  This class depends on CustomNets.dataload_ for loading data. """
    def __init__(self, PTS_BASE, nP, nN, batch_size=4, n_samples=500, initial_epoch=0, refresh_data_after_n_epochs=20, image_nrows=240, image_ncols=320, image_nchnl=1 ):

        # assert( type(n_samples) == type(()) )
        self.n_samples_pitts = int(n_samples)
        self.epoch = initial_epoch
        self.batch_size = int(batch_size)
        self.refresh_data_after_n_epochs = int(refresh_data_after_n_epochs)
        self.nP = nP
        self.nN = nN
        self.ENABLE_IMSHOW = True
        # self.n_samples = n_samples
        print tcolor.OKGREEN, '-------------PitsSequence Config--------------', tcolor.ENDC
        print 'n_samples  : ', self.n_samples_pitts
        print 'batch_size : ', self.batch_size
        print 'refresh_data_after_n_epochs : ', self.refresh_data_after_n_epochs
        print 'image_nrows: ', image_nrows, '\timage_ncols: ', image_ncols, '\timage_nchnl: ', image_nchnl
        print '# positive samples (nP) = ', self.nP
        print '# negative samples (nP) = ', self.nN
        print tcolor.OKGREEN, '----------------------------------------------', tcolor.ENDC


        self.resize = (image_ncols, image_nrows)
        if image_nchnl == 3:
            self.return_gray = False
        else :
            self.return_gray = True


        # PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
        self.pr = PittsburgRenderer( PTS_BASE )
        self.D = self.pr.step_n_times(n_samples=self.n_samples_pitts, nP=nP, nN=nN, resize=self.resize, return_gray=self.return_gray, ENABLE_IMSHOW=self.ENABLE_IMSHOW )
        print 'len(D)=', len(self.D), '\tD[0].shape=', self.D[0].shape
        self.y = np.zeros( len(self.D) )
        # self.steps = int(np.ceil(len(self.D) / float(self.batch_size)))



    def __len__(self):
        # print '-----len=====', int(np.ceil(len(self.D) / float(self.batch_size)))
        return int(np.ceil(len(self.D) / float(self.batch_size)))

    def __getitem__(self, idx):
        if idx == 0:
            self.on_epoch_start()
        batch_x = self.D[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # return np.array( batch_x ), np.array( batch_y )
        # return np.array( batch_x )*1./255. - 0.5, np.array( batch_y )
        return np.array( batch_x ).astype('float32')*2./255. - 1.0, np.array( batch_y )
       #TODO: Can return another number (sample_weight) for the sample. Which can be judge say by GMS matcher. If we see higher matches amongst +ve set ==> we have good positive samples,


    def on_epoch_end(self):
        N = self.refresh_data_after_n_epochs
        #print '-----on_epoch_start, self.epoch=', self.epoch, '\tself.refresh_data_after_n_epochs=', self.refresh_data_after_n_epochs

        if self.epoch % N == 0 and self.epoch > 0 :
            print '[on_epoch_start] done %d epochs, so load new data\t' %(N), int_logr.dir()
            # Sample Data
            # self.D = dataload_( n_tokyoTimeMachine=self.n_samples_tokyo, n_Pitssburg=self.n_samples_pitts, nP=nP, nN=nN )

            self.D = self.pr.step_n_times(n_samples=self.n_samples_pitts, nP=self.nP, nN=self.nN, resize=self.resize, return_gray=self.return_gray, ENABLE_IMSHOW=self.ENABLE_IMSHOW )
            self.y = np.zeros( len(self.D) )
            print 'epoch=', self.epoch, '\tlen(D)=', len(self.D), '\tD[0].shape=', self.D[0].shape


            if self.epoch > 10:
            # if self.epoch > 400 and self.n_samples_pitts<0:
                # Data Augmentation after 400 epochs.
                print tcolor.WARNING, 'do_typical_data_aug', tcolor.ENDC
                self.D = do_typical_data_aug( self.D )
            else:
                print tcolor.WARNING, 'NO Data Augmentation', tcolor.ENDC


            print 'returned len(self.D)=', len(self.D), 'self.D[0].shape=', self.D[0].shape
            self.y = np.zeros( len(self.D) )
            # modify data
        self.epoch += 1






class CustomModelCallback(keras.callbacks.Callback):
    def __init__(self, model_tosave, int_logr, save_model_every_n_epochs=100 ):
        self.m_model = model_tosave
        self.m_int_logr = int_logr
        self.save_model_every_n_epochs = save_model_every_n_epochs

    def on_epoch_begin(self, epoch, logs={}):
        if epoch>0 and epoch%self.save_model_every_n_epochs == 0:
            # TODO: Eventually get rid of this. The current recommend way is to do away with json and store arch and weights together in a HDF5 file
            #fname = self.m_int_logr.dir() + '/core_model.%d.keras' %(epoch)
            #print 'CustomModelCallback::Save Intermediate Model : ', fname
            #self.m_model.save( fname )

            fname_h5 = self.m_int_logr.dir() + '/modelarch_and_weights.%d.h5' %(epoch)
            print 'CustomModelCallback::Save Intermediate Model : ', fname_h5
            self.m_model.save( fname_h5 )

        if epoch%5 == 0:
            print 'CustomModelCallback::m_int_logr=', self.m_int_logr.dir()

import signal
import sys
def signal_handler(sig, frame):
    #TODO: somehow get access to current epoch number and write the mdoel file accordingly
    print('You pressed Ctrl+C!')
    print 'Save Current Model : ',  int_logr.dir() + '/core_modelX.keras'
    # model.save( int_logr.dir() + '/core_modelX.keras' )
    model.save( int_logr.dir() + '/modelarch_and_weights.X.h5' )
    sys.exit(0)

# Training
if __name__ == '__main__':
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    print '==================================================='
    print 'Keras version: ', keras.__version__
    print 'Tensorflow Version: ', tf.__version__
    print '==================================================='
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))
    signal.signal(signal.SIGINT, signal_handler)

    #---
    image_nrows = 240
    image_ncols = 320
    image_nchnl = 1 #cannot make this to 1 if u want to use VGG-pretained-weights (from imagenet), bcoz imagenet was trained with color images. However this is usable (can set chanls=1) if you want to train from scratch

    #---
    #   note: some other thing thats need to be set
    #      a. CNN Type: VGG16, mobilenet
    #      b. base CNN layer

    CNN_type =  'mobilenetv2'       #'mobilenet', 'vgg16', 'mobilenetv2'
    layer_name= 'block_9_add' #'conv_pw_7_relu', 'block5_pool', 'block_9_add'
    init_model_weights = None #'imagenet', None. imagenet only nchanls=3

    nP = 6
    nN = 6
    netvlad_num_clusters = 16

    initial_epoch = 0 # for resuming training. If this is a non-zero value will load the corresponding model from log_dir
    n_samples = 500 #< training sames to load for the epoch
    batch_size = 4  #< training batch size. aka, how many tuples must be there in 1 sample.
    refresh_data_after_n_epochs = 20  # after how many iterations you want me to refresh the data
    save_model_every_n_epochs = 100 # after every how many epochs you want me to write the models


    #---
    # LOG_DIR = './models.keras/mobilenet_conv7_allpairloss/'
    # LOG_DIR = './models.keras/mobilenet_conv7_tripletloss2/'

    # LOG_DIR = './models.keras/mobilenet_conv7_quash_chnls_allpairloss/'
    # LOG_DIR = './models.keras/mobilenet_conv7_quash_chnls_tripletloss2_K64/'

    # LOG_DIR = './models.keras/vgg16/block5_pool_k32_allpairloss'
    # LOG_DIR = './models.keras/vgg16/block5_pool_k32_tripletloss2'

    # LOG_DIR = './models.keras/mobilenet_new/pw13_quash_chnls_k16_allpairloss'
    # LOG_DIR = './models.keras/mobilenet_new/pw13_quash_chnls_k16_tripletloss2'

    # LOG_DIR = './models.keras/tmp1/'
    # LOG_DIR = './models.keras/Apr2019/gray_conv6_K16__centeredinput'
    # LOG_DIR = './models.keras/Apr2019/gray_conv6_K16Ghost1__centeredinput'
    # LOG_DIR = './models.keras/Apr2019/color_conv6_K16Ghost1__centeredinput'
    # LOG_DIR = './models.keras/May2019/centeredinput-gray__mobilenet-conv7__K16__allpairloss'

    LOG_DIR = './models.keras/June2019/centeredinput-m1to1-%dx%dx%d__%s-%s__K%d__allpairloss' %(image_nrows, image_ncols, image_nchnl, CNN_type, layer_name, netvlad_num_clusters)
    global int_logr
    int_logr = InteractiveLogger( LOG_DIR )

    #---
    PITS_PATH = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
    PITS_VAL_PATH = '/Bulk_Data/data_Akihiko_Torii/Pitssburg_validation/'




    #--------------------------------------------------------------------------
    # Core Model Setup
    #--------------------------------------------------------------------------
    # Build
    global model
    input_img = keras.layers.Input( shape=(image_nrows, image_ncols, image_nchnl ) )

    if CNN_type=='mobilenet' or CNN_type=='vgg16' or CNN_type =='mobilenetv2':
        pass
    else:
        assert( False )

    # weights=imagenet will only work with nchanls=3
    cnn = None
    if CNN_type=='mobilenet':
        cnn = make_from_mobilenet( input_img, layer_name=layer_name,\
                    weights=init_model_weights, kernel_regularizer=keras.regularizers.l2(0.01) )

    if CNN_type == 'vgg16':
        cnn = make_from_vgg16( input_img, weights=init_model_weights, layer_name=layer_name, kernel_regularizer=keras.regularizers.l2(0.001) )

    if CNN_type == 'mobilenetv2':
        cnn = make_from_mobilenetv2( input_img,  layer_name=layer_name,\
        weights=init_model_weights, kernel_regularizer=keras.regularizers.l2(0.01) )


    # Reduce nChannels of the output.
    # @ Downsample (Optional)
    if False: #Downsample last layer (Reduce nChannels of the output.)
        cnn_dwn = keras.layers.Conv2D( 256, (1,1), padding='same', activation='relu' )( cnn )
        cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
        cnn_dwn = keras.layers.Conv2D( 64, (1,1), padding='same', activation='relu' )( cnn_dwn )
        cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
        cnn = cnn_dwn

    # out, out_amap = NetVLADLayer(num_clusters = 32)( cnn )
    out = NetVLADLayer(num_clusters = netvlad_num_clusters)( cnn )
    # out = GhostVLADLayer(num_clusters = netvlad_num_clusters, num_ghost_clusters=1)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=out )

    # Plot
    model.summary()
    keras.utils.plot_model( model, to_file=int_logr.dir()+'/core.png', show_shapes=True )
    int_logr.save_model_as_json( 'model.json', model )
        # look at test_loadjsonmodel.py for demo on how to load model with json file.




    if initial_epoch > 0:
        # Load Previous Weights
        model.load_weights(  int_logr.dir() + '/core_model.%d.keras' %(initial_epoch) )




    #--------------------------------------------------------------------------
    # TimeDistributed - This is a hack to get Weakly-supervised learning working
    #--------------------------------------------------------------------------
    t_input = keras.layers.Input( shape=(1+nP+nN, image_nrows, image_ncols, image_nchnl ) )
    t_out = keras.layers.TimeDistributed( model )( t_input )

    t_model = keras.models.Model( inputs=t_input, outputs=t_out )

    t_model.summary()
    keras.utils.plot_model( t_model, to_file=int_logr.dir()+'/core_t.png', show_shapes=True )
    print 'Write Directory : ', int_logr.dir()
    # int_logr.fire_editor()


    #--------------------------------------------------------------------------
    # Compile
    #--------------------------------------------------------------------------

    #-----(a) Gradient Descent Algorithm
    # sgdopt = keras.optimizers.Adadelta( lr=2.0 )
    sgdopt = keras.optimizers.Adam(  )

    #-----(b) Loss
    loss = triplet_loss2_maker(nP=nP, nN=nN, epsilon=0.3)
    # loss = allpair_hinge_loss_with_positive_set_deviation_maker(nP=nP, nN=nN, epsilon=0.3, opt_lambda=0.5 )
    # loss = allpair_hinge_loss_maker( nP=nP, nN=nN, epsilon=0.3 )

    #-----(c) Evaluation Metric (on validation_data)
    metrics = [ allpair_count_goodfit_maker( nP=nP, nN=nN, epsilon=0.3 ),
                positive_set_deviation_maker(nP=nP, nN=nN)
              ]

    # t_model.compile( loss=allpair_hinge_loss, optimizer=sgdopt, metrics=[allpair_count_goodfit] )
    t_model.compile( loss=loss, optimizer=sgdopt, metrics=metrics )


    #-----(d) Callbacks Setup
    import tensorflow as tf
    tb = tf.keras.callbacks.TensorBoard( log_dir=int_logr.dir() )
    saver_cb = CustomModelCallback( model, int_logr, save_model_every_n_epochs=save_model_every_n_epochs )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6, patience=75, verbose=1, min_lr=0.05)


    #-----(e) Sequence generator setup
    gen = PitsSequence( PITS_PATH,nP=nP, nN=nN, batch_size=batch_size, n_samples=n_samples, initial_epoch=initial_epoch, refresh_data_after_n_epochs=refresh_data_after_n_epochs, image_nrows=image_nrows, image_ncols=image_ncols, image_nchnl=image_nchnl)
    gen_validation = PitsSequence(PITS_VAL_PATH, nP=nP, nN=nN, batch_size=batch_size, n_samples=n_samples, refresh_data_after_n_epochs=refresh_data_after_n_epochs, image_nrows=image_nrows, image_ncols=image_ncols, image_nchnl=image_nchnl )
    # code.interact(local=locals())

    #-----(f)  fit_generator() ----#
    history = t_model.fit_generator( generator = gen,
                            epochs=2200, verbose=1, initial_epoch=initial_epoch,
                            validation_data = gen_validation ,
                            callbacks=[tb,saver_cb,reduce_lr],
                            use_multiprocessing=False, workers=0,
                         )
    print 'Save Final Model : ',  int_logr.dir() + '/core_model.keras'
    model.save( int_logr.dir() + '/core_model.keras' )
    print 'Save Final Model : ',  int_logr.dir() + '/modelarch_and_weights.h5'
    model.save( int_logr.dir() + '/modelarch_and_weights.h5' )
