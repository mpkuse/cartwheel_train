import keras
import json
import pprint
import numpy as np
import cv2
import code

from CustomNets import NetVLADLayer

#--------------------------- UTILS --------------------------------------------#
def open_json_file( fname ):
    print 'Load JSON file: ', fname
    jsonX = json.loads(open(fname).read())
    return jsonX


def change_model_inputshape(model, new_input_shape=(None, 40, 40, 3), verbose=False):
    """
    Given a model and new input shape it changes all the allocations.

    Note: It uses custom_objects={'NetVLAD': NetVLADLayer}. If you have any other
    custom-layer change the code here accordingly. 
    """
    # replace input shape of first layer
    model._layers[0].batch_input_shape = new_input_shape

    # feel free to modify additional parameters of other layers, for example...
    # model._layers[2].pool_size = (8, 8)
    # model._layers[2].strides = (8, 8)

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json(), custom_objects={'NetVLADLayer': NetVLADLayer} )
    new_model.summary()

    # copy weights from old model to new one
    print 'copy weights from old model to new one....this usually takes upto 10 sec'
    for layer in new_model.layers:
        try:
            if verbose:
                print( 'transfer weights for layer.name='+layer.name )
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # test new model on a random input image
    # X = np.random.rand(new_input_shape[0], new_input_shape[1], new_input_shape[2], new_input_shape[3] )
    # y_pred = new_model.predict(X)
    # print('try predict with a random input_img with shape='+str(X.shape)+ str(y_pred) )

    return new_model

#--------------------------- END UTILS ----------------------------------------#
