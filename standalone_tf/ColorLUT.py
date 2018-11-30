## Class for lookup colors. 64 colors different looking to color cluster assgnment.
import cv2
import numpy as np

class ColorLUT:
    def __init__(self):
        _color = []
        _color.append( [0, 0, 0] )
        _color.append( [0, 255, 0] )
        _color.append( [0, 0, 255] )
        _color.append( [255, 0, 0] )
        _color.append( [1, 255, 254] )
        _color.append( [255, 166, 254] )
        _color.append( [255, 219, 102] )
        _color.append( [0, 100, 1] )
        _color.append( [1, 0, 103] )
        _color.append( [149, 0, 58] )
        _color.append( [0, 125, 181] )
        _color.append( [255, 0, 246] )
        _color.append( [255, 238, 232] )
        _color.append( [119, 77, 0] )
        _color.append( [144, 251, 146] )
        _color.append( [0, 118, 255] )
        _color.append( [213, 255, 0] )
        _color.append( [255, 147, 126] )
        _color.append( [106, 130, 108] )
        _color.append( [255, 2, 157] )
        _color.append( [254, 137, 0] )
        _color.append( [122, 71, 130] )
        _color.append( [126, 45, 210] )
        _color.append( [133, 169, 0] )
        _color.append( [255, 0, 86] )
        _color.append( [164, 36, 0] )
        _color.append( [0, 174, 126] )
        _color.append( [104, 61, 59] )
        _color.append( [189, 198, 255] )
        _color.append( [38, 52, 0] )
        _color.append( [189, 211, 147] )
        _color.append( [0, 185, 23] )
        _color.append( [158, 0, 142] )
        _color.append( [0, 21, 68] )
        _color.append( [194, 140, 159] )
        _color.append( [255, 116, 163] )
        _color.append( [1, 208, 255] )
        _color.append( [0, 71, 84] )
        _color.append( [229, 111, 254] )
        _color.append( [120, 130, 49] )
        _color.append( [14, 76, 161] )
        _color.append( [145, 208, 203] )
        _color.append( [190, 153, 112] )
        _color.append( [150, 138, 232] )
        _color.append( [187, 136, 0] )
        _color.append( [67, 0, 44] )
        _color.append( [222, 255, 116] )
        _color.append( [0, 255, 198] )
        _color.append( [255, 229, 2] )
        _color.append( [98, 14, 0] )
        _color.append( [0, 143, 156] )
        _color.append( [152, 255, 82] )
        _color.append( [117, 68, 177] )
        _color.append( [181, 0, 255] )
        _color.append( [0, 255, 120] )
        _color.append( [255, 110, 65] )
        _color.append( [0, 95, 57] )
        _color.append( [107, 104, 130] )
        _color.append( [95, 173, 78] )
        _color.append( [167, 87, 64] )
        _color.append( [165, 255, 210] )
        _color.append( [255, 177, 103] )
        _color.append( [0, 155, 255] )
        _color.append( [232, 94, 190] )
        for i in range(64,256):
        	_color.append( [0,0,0] )
        _color = np.array( _color ).astype( 'uint8')
        self._color = _color

    def lut( self, im ):
        im = np.array( im )
        R = (self._color[:,0])[ im ]
        G = (self._color[:,1])[ im ]
        B = (self._color[:,2])[ im ]
        lut = np.dstack( (B,G,R) )  #opencv array now
        return np.array(lut)

    def get_color( self, i ):
        return self._color[i].astype('int32')
