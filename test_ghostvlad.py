import keras
import numpy as np
import code
from CustomNets import NetVLADLayer, GhostVLADLayer



input_img = keras.layers.Input( shape=(60, 80, 256 ) )
out = GhostVLADLayer(num_clusters = 16, num_ghost_clusters = 1)( input_img )
model = keras.models.Model( inputs=input_img, outputs=out )

model.predict( np.random.rand( 1,60,80,256).astype('float32') )
