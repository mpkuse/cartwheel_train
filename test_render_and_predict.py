import keras
import json
import pprint
import numpy as np
import cv2
import code

from CustomNets import NetVLADLayer
from PittsburgRenderer import PittsburgRenderer
from predict_utils import open_json_file, change_model_inputshape


def add_image_caption( im, txt ) :

    caption_im_width = 30*len( str(txt).split( ';' ) )

    if len(im.shape) == 3:
        zer = np.zeros( [caption_im_width, im.shape[1], im.shape[2]], dtype='uint8' )
    else:
        if len(im.shape) == 2:
            zer = np.zeros( [caption_im_width, im.shape[1]], dtype='uint8' )
        else:
            assert( False )


    for e, tx in enumerate( str(txt).split( ';' ) ):
        zer = cv2.putText(zer, str(tx), (3,20+30*e), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255) )
    return np.concatenate( [im, zer] )


def imshow_set( win_name, D, caption=None ):
    """ D: Nx320x240x3 or Nx320x240x1. 320 and 240 can be in general any values. Basically shows
    multile images
    """
    assert( len(D.shape) == 4 )
    hcat = np.concatenate( D.astype('uint8'), axis=1)
    if caption is not None:
        hcat = add_image_caption( hcat, caption )

    cv2.imshow( str(win_name), hcat )


def render_and_predict():
    #---------- Setup model
    LOG_DIR = 'models.keras/Apr2019/gray_conv6_K16__centeredinput/'

    # Load json-model
    json_model_fname = LOG_DIR+'/model.json'
    print 'Load json-model: ', json_model_fname
    json_string = open_json_file( json_model_fname )
    model = keras.models.model_from_json(str(json_string),  custom_objects={'NetVLADLayer': NetVLADLayer} )

    # Load Weights
    model_fname = LOG_DIR+'/core_model.%d.keras' %(400)
    print 'Load model: ', model_fname
    model.load_weights(  model_fname )

    # change_model_inputshape
    new_model = change_model_inputshape( model, new_input_shape=(11,480,640,1) )



    #--------------- Setup renderer
    PITS_VAL_PATH = '/Bulk_Data/data_Akihiko_Torii/Pitssburg_validation/'
    pr = PittsburgRenderer( PITS_VAL_PATH )

    for _ in range(100):
        print '---'
        a,b = pr.step(nP=5, nN=5, ENABLE_IMSHOW=False, return_gray=True, resize=(640, 480))
        a = np.copy(a) #if you dont do a copy, keras.predict gives sigsegv

        # from CustomNets import do_typical_data_aug
        # a = do_typical_data_aug( a )





        # Predict
        print 'a.shape=', a.shape, '\ta.dtype=', a.dtype
        a_pred = new_model.predict( (a.astype('float32')-128.)/255. )
        # a0_pred = new_model.predict(  np.expand_dims(a[0],0).astype('float32')  )
        # a_p0_pred = new_model.predict( np.expand_dims(a[1],0).astype('float32'))
        # a_n0_pred = new_model.predict(np.expand_dims(a[6],0).astype('float32') )
        DOT = np.matmul( a_pred, a_pred[0] )
        print '<a, a0> ', DOT


        # imshow
        cv2.imshow( 'query', a[0].astype('uint8') )
        imshow_set( 'sim', a[1:6], str(DOT[1:6]) )
        imshow_set( 'dif', a[6:], str(DOT[6:]) )

        key = cv2.waitKey(0)
        if key == ord('q'):
            print 'Quit...'
            break

    code.interact( local=locals() )



if __name__ == '__main__':
    render_and_predict()
