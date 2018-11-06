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


# CustomNets
from CustomNets import NetVLADLayer
from CustomNets import dataload_, do_typical_data_aug
from CustomNets import make_from_mobilenet

# CustomLoses
from CustomLosses import triplet_loss2_maker, allpair_hinge_loss_maker, allpair_count_goodfit_maker, positive_set_deviation_maker, allpair_hinge_loss_with_positive_set_deviation_maker

from InteractiveLogger import InteractiveLogger


# Data
from TimeMachineRender import TimeMachineRender
# from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer

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


            if self.epoch > 400:
                # Data Augmentation after 400 epochs
                self.D = do_typical_data_aug( self.D )

            print 'dataload_ returned len(self.D)=', len(self.D), 'self.D[0].shape=', self.D[0].shape
            self.y = np.zeros( len(self.D) )
            # modify data
        self.epoch += 1






# Training
if __name__ == '__main__':
    image_nrows = 240
    image_ncols = 320
    image_nchnl = 3
    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_allpairloss/' )
    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_tripletloss2/' )

    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_quash_chnls_allpairloss/' )
    # int_logr = InteractiveLogger( './models.keras/mobilenet_conv7_quash_chnls_tripletloss2_K64/' )
    int_logr = InteractiveLogger( './models.keras/test/test2' )
    nP = 6
    nN = 6

    #--------------------------------------------------------------------------
    # Core Model Setup
    #--------------------------------------------------------------------------
    # Build
    input_img = keras.layers.Input( shape=(image_nrows, image_ncols, image_nchnl ) )
    cnn = make_from_mobilenet( input_img )
    # Reduce nChannels of the output.
    # cnn_dwn = keras.layers.Conv2D( 256, (1,1), padding='same', activation='relu' )( cnn )
    # cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
    # cnn_dwn = keras.layers.Conv2D( 32, (1,1), padding='same', activation='relu' )( cnn_dwn )
    # cnn_dwn = keras.layers.normalization.BatchNormalization()( cnn_dwn )
    # cnn = cnn_dwn

    out, out_amap = NetVLADLayer(num_clusters = 16)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=out )

    # Plot
    model.summary()
    keras.utils.plot_model( model, to_file=int_logr.dir()+'/core.png', show_shapes=True )
    int_logr.add_file( 'model.json', model.to_json() )


    # Load Previous Weights
    # model.load_weights(  int_logr.dir() + '/core_model.950.keras' )
    initial_epoch = 0




    #--------------------------------------------------------------------------
    # TimeDistributed
    #--------------------------------------------------------------------------
    t_input = keras.layers.Input( shape=(1+nP+nN, image_nrows, image_ncols, image_nchnl ) )
    t_out = keras.layers.TimeDistributed( model )( t_input )

    t_model = keras.models.Model( inputs=t_input, outputs=t_out )

    t_model.summary()
    keras.utils.plot_model( t_model, to_file=int_logr.dir()+'core_t.png', show_shapes=True )
    print 'Write Directory : ', int_logr.dir()
    # int_logr.fire_editor()


    #--------------------------------------------------------------------------
    # Compile
    #--------------------------------------------------------------------------
    sgdopt = keras.optimizers.Adadelta(  )

    # loss = triplet_loss2_maker(nP=nP, nN=nN, epsilon=0.3)
    loss = allpair_hinge_loss_with_positive_set_deviation_maker(nP=nP, nN=nN, epsilon=0.3, opt_lambda=0.1 )

    metrics = [ allpair_count_goodfit_maker( nP=nP, nN=nN, epsilon=0.3 ),
                positive_set_deviation_maker(nP=nP, nN=nN)
              ]

    # t_model.compile( loss=allpair_hinge_loss, optimizer=sgdopt, metrics=[allpair_count_goodfit] )
    t_model.compile( loss=loss, optimizer=sgdopt, metrics=metrics )


    import tensorflow as tf
    tb = tf.keras.callbacks.TensorBoard( log_dir=int_logr.dir() )


    t_model.fit_generator( generator=WSequence(nP, nN, n_samples=(500,-1), initial_epoch=initial_epoch),
                            epochs=1200, verbose=1, initial_epoch=initial_epoch,
                            validation_data = WSequence(nP, nN, n_samples=(-1,500) ),
                            callbacks=[tb]
                         )
    print 'Save Final Model : ',  int_logr.dir() + '/core_model.keras'
    model.save( int_logr.dir() + '/core_model.keras' )
