""" NetVLAD training. Using features from deeper VGG layers.

    - Input -- BN -- MobileNet -- NetVLAD
    - Deeper features (instead of currently using very shallow ones)
    - Verify my pairwise_loss also implement triplet loss
    - Keras sequence use

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
from CustomNets import NetVLADLayer
from CustomNets import dataload_, do_typical_data_aug
from CustomNets import make_from_mobilenet, make_from_vgg16

# CustomLoses
from CustomLosses import triplet_loss2_maker, allpair_hinge_loss_maker, allpair_count_goodfit_maker, positive_set_deviation_maker, allpair_hinge_loss_with_positive_set_deviation_maker

from InteractiveLogger import InteractiveLogger
import TerminalColors
tcolor = TerminalColors.bcolors()

# Data
from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer


# TODO : removal
class WSequence(keras.utils.Sequence):
    """  This class depends on CustomNets.dataload_ for loading data. """
    def __init__(self, nP, nN, n_samples=(500,-1), initial_epoch=0 ):

        assert( type(n_samples) == type(()) )
        assert( len(n_samples) == 2 )
        self.n_samples_tokyo = n_samples[0]
        self.n_samples_pitts = n_samples[1]
        self.epoch = initial_epoch
        self.batch_size = 4
        self.refresh_data_after_n_epochs = 20
        # self.n_samples = n_samples


        # This will load the data
        self.D = dataload_( n_tokyoTimeMachine=self.n_samples_tokyo, n_Pitssburg=self.n_samples_pitts, nP=nP, nN=nN )
        print 'dataload_ returned len(self.D)=', len(self.D), 'self.D[0].shape=', self.D[0].shape
        self.y = np.zeros( len(self.D) )



    def __len__(self):
        return int(np.ceil(len(self.D) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.D[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array( batch_x ), np.array( batch_y )
       #TODO: Can return another number (sample_weight) for the sample. Which can be judge say by GMS matcher. If we see higher matches amongst +ve set ==> we have good positive samples,


    def on_epoch_end(self):
        N = self.refresh_data_after_n_epochs

        if self.epoch % N == 0 and self.epoch > 0 :
            print '[on_epoch_end] done %d epochs, so load new data\t' %(N), int_logr.dir()
            # Sample Data
            self.D = dataload_( n_tokyoTimeMachine=self.n_samples_tokyo, n_Pitssburg=self.n_samples_pitts, nP=nP, nN=nN )



            # if self.epoch > 400:
            if self.epoch > 400 and self.n_samples_pitts<0:
                # Data Augmentation after 400 epochs. Only do for Tokyo which are used for training. ie. dont augment Pitssburg.
                self.D = do_typical_data_aug( self.D )

            print 'dataload_ returned len(self.D)=', len(self.D), 'self.D[0].shape=', self.D[0].shape
            self.y = np.zeros( len(self.D) )
            # modify data
        self.epoch += 1


# TODO This is a simpler implementation of WSequence. Eventually delete WSequence
class PitsSequence(keras.utils.Sequence):
    """  This class depends on CustomNets.dataload_ for loading data. """
    def __init__(self, PTS_BASE, nP, nN, n_samples=500, initial_epoch=0 ):

        # assert( type(n_samples) == type(()) )
        self.n_samples_pitts = int(n_samples)
        self.epoch = initial_epoch
        self.batch_size = 4
        self.refresh_data_after_n_epochs = 20
        self.nP = nP
        self.nN = nN
        # self.n_samples = n_samples
        print tcolor.OKGREEN, '-------------PitsSequence Config--------------', tcolor.ENDC
        print 'n_samples  : ', self.n_samples_pitts
        print 'batch_size : ', self.batch_size
        print 'refresh_data_after_n_epochs : ', self.refresh_data_after_n_epochs
        print tcolor.OKGREEN, '----------------------------------------------', tcolor.ENDC


        # This will load the data
        # self.D = dataload_( n_tokyoTimeMachine=self.n_samples_tokyo, n_Pitssburg=self.n_samples_pitts, nP=nP, nN=nN )
        # print 'dataload_ returned len(self.D)=', len(self.D), 'self.D[0].shape=', self.D[0].shape
        # self.y = np.zeros( len(self.D) )


        # PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
        self.pr = PittsburgRenderer( PTS_BASE )
        self.D = self.pr.step_n_times(n_samples=500, nP=nP, nN=nN, resize=(320,240), return_gray=True, ENABLE_IMSHOW=True )
        print 'len(D)=', len(self.D), '\tD[0].shape=', self.D[0].shape
        self.y = np.zeros( len(self.D) )



    def __len__(self):
        return int(np.ceil(len(self.D) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.D[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # return np.array( batch_x ), np.array( batch_y )
        return np.array( batch_x )*1./255. - 0.5, np.array( batch_y )
       #TODO: Can return another number (sample_weight) for the sample. Which can be judge say by GMS matcher. If we see higher matches amongst +ve set ==> we have good positive samples,


    def on_epoch_end(self):
        N = self.refresh_data_after_n_epochs

        if self.epoch % N == 0 and self.epoch > 0 :
            print '[on_epoch_end] done %d epochs, so load new data\t' %(N), int_logr.dir()
            # Sample Data
            # self.D = dataload_( n_tokyoTimeMachine=self.n_samples_tokyo, n_Pitssburg=self.n_samples_pitts, nP=nP, nN=nN )

            self.D = self.pr.step_n_times(n_samples=500, nP=self.nP, nN=self.nN, resize=(320,240), return_gray=True, ENABLE_IMSHOW=True )
            print 'len(D)=', len(self.D), '\tD[0].shape=', self.D[0].shape


            # if self.epoch > 400:
            if self.epoch > 400 and self.n_samples_pitts<0:
                # Data Augmentation after 400 epochs. Only do for Tokyo which are used for training. ie. dont augment Pitssburg.
                self.D = do_typical_data_aug( self.D )

            print 'dataload_ returned len(self.D)=', len(self.D), 'self.D[0].shape=', self.D[0].shape
            self.y = np.zeros( len(self.D) )
            # modify data
        self.epoch += 1



class CustomModelCallback(keras.callbacks.Callback):
    def __init__(self, model_tosave, int_logr ):
        self.m_model = model_tosave
        self.m_int_logr = int_logr

    def on_epoch_begin(self, epoch, logs={}):
        if epoch>0 and epoch%200 == 0:
            fname = self.m_int_logr.dir() + '/core_model.%d.keras' %(epoch)
            print 'Save Intermediate Model : ', fname
            self.m_model.save( fname )

        if epoch%5 == 0:
            print 'm_int_logr=', self.m_int_logr.dir()




# Training
if __name__ == '__main__':
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

    image_nrows = 240
    image_ncols = 320
    image_nchnl = 3

    nP = 6
    nN = 6
    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_allpairloss/' )
    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_tripletloss2/' )

    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_quash_chnls_allpairloss/' )
    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_quash_chnls_tripletloss2_K64/' )

    # int_logr = InteractiveLogger( './models.keras/vgg16/block5_pool_k32_allpairloss' )
    # int_logr = InteractiveLogger( './models.keras/vgg16/block5_pool_k32_tripletloss2' )

    # int_logr = InteractiveLogger( './models.keras/mobilenet_new/pw13_quash_chnls_k16_allpairloss' )
    # int_logr = InteractiveLogger( './models.keras/mobilenet_new/pw13_quash_chnls_k16_tripletloss2' )

    int_logr = InteractiveLogger( './models.keras/tmp_staticnormalized_images/' )


    #--------------------------------------------------------------------------
    # Core Model Setup
    #--------------------------------------------------------------------------
    # Build
    input_img = keras.layers.Input( shape=(image_nrows, image_ncols, image_nchnl ) )
    # cnn = make_from_mobilenet( input_img, layer_name='conv_pw_13_relu', kernel_regularizer=None )
    cnn = make_from_vgg16( input_img, layer_name='block5_pool' )
    # Reduce nChannels of the output.
    # @ Downsample (Optional)
    if False: #Downsample last layer (Reduce nChannels of the output.)
        cnn_dwn = keras.layers.Conv2D( 256, (1,1), padding='same', activation='relu' )( cnn )
        cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
        cnn_dwn = keras.layers.Conv2D( 64, (1,1), padding='same', activation='relu' )( cnn_dwn )
        cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
        cnn = cnn_dwn

    out, out_amap = NetVLADLayer(num_clusters = 32)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=out )

    # Plot
    model.summary()
    keras.utils.plot_model( model, to_file=int_logr.dir()+'/core.png', show_shapes=True )
    int_logr.add_file( 'model.json', model.to_json() )


    initial_epoch = 0
    if initial_epoch > 0:
        # Load Previous Weights
        model.load_weights(  int_logr.dir() + '/core_model.%d.keras' %(initial_epoch) )




    #--------------------------------------------------------------------------
    # TimeDistributed
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
    sgdopt = keras.optimizers.Adadelta( )
    # sgdopt = keras.optimizers.Adam(  )

    loss = triplet_loss2_maker(nP=nP, nN=nN, epsilon=0.3)
    # loss = allpair_hinge_loss_with_positive_set_deviation_maker(nP=nP, nN=nN, epsilon=0.3, opt_lambda=0.5 )
    # loss = allpair_hinge_loss_maker( nP=nP, nN=nN, epsilon=0.3 )

    metrics = [ allpair_count_goodfit_maker( nP=nP, nN=nN, epsilon=0.3 ),
                positive_set_deviation_maker(nP=nP, nN=nN)
              ]

    # t_model.compile( loss=allpair_hinge_loss, optimizer=sgdopt, metrics=[allpair_count_goodfit] )
    t_model.compile( loss=loss, optimizer=sgdopt, metrics=metrics )


    import tensorflow as tf
    tb = tf.keras.callbacks.TensorBoard( log_dir=int_logr.dir() )
    saver_cb = CustomModelCallback( model, int_logr )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6, patience=75, verbose=1, min_lr=0.1)


    history = t_model.fit_generator( generator=PitsSequence('/Bulk_Data/data_Akihiko_Torii/Pitssburg/' ,nP=nP, nN=nN, n_samples=500, initial_epoch=initial_epoch),
                            epochs=2200, verbose=1, initial_epoch=initial_epoch,
                            validation_data = PitsSequence('/Bulk_Data/data_Akihiko_Torii/Pitssburg_validation/', nP=nP, nN=nN, n_samples=500 ),
                            callbacks=[tb,saver_cb,reduce_lr]
                         )
    print 'Save Final Model : ',  int_logr.dir() + '/core_model.keras'
    model.save( int_logr.dir() + '/core_model.keras' )

    # print 'Save Json : ', int_logr.dir()+'/history.json'
    # with open( int_logr.dir()+'/history.json', 'w') as f:
    #     json.dump(history.history, f)
    #
    #
    # print 'Save History : ', int_logr.dir()+'/history.pickle'
    # with open( int_logr.dir()+'/history.pickle', 'wb' ) as handle:
    #     pickle.dump(history, handle )
