## A demo of loading the json model

import keras
import json
from CustomNets import NetVLADLayer

def open_json_file( fname ):
    print 'Load JSON file: ', fname
    jsonX = json.loads(open(fname).read())
    return jsonX



PATH = 'models.keras/tmp/'
json_string = open_json_file( PATH+'/model.json' )
model = keras.models.model_from_json(str(json_string),  custom_objects={'NetVLADLayer': NetVLADLayer} )
