#!/usr/bin/env python

''' The Census Transform

    Scan an 8 bit greyscale image with a 3x3 window
        At each scan position create an 8 bit number by comparing the value
            of the centre pixel in the 3x3 window with that of its 8 neighbours.
                The bit is set to 1 if the outer pixel >= the centre pixel

                    See http://stackoverflow.com/questions/38265364/census-transform-in-python-opencv

                        Written by PM 2Ring 2016.07.09
                        '''

import numpy as np
from PIL import Image
import time
import cv2


def censusTransformSingleChannel(src_bytes):
    h,w = src_bytes.shape
#Initialize output array
    census = np.zeros((h-2, w-2), dtype='uint8')
    #census1 = np.zeros((h, w), dtype='uint8')

#centre pixels, which are offset by (1, 1)
    cp = src_bytes[1:h-1, 1:w-1]

#offsets of non-central pixels 
    offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

#Do the pixel comparisons
    for u,v in offsets:
        census = (census << 1) | (src_bytes[v:v+h-2, u:u+w-2] >= cp)

    return census


def censusTransform( src_bytes ):
    # chk num of channels. if 1 call as is, if 3 call it with each channel
    if len(src_bytes.shape) == 2: #single channel
        census = censusTransformSingleChannel( np.lib.pad( src_bytes, 1, 'constant', constant_values=0 ) )

    if len( src_bytes.shape ) == 3 and src_bytes.shape[2] == 3 : 
        census_a = censusTransformSingleChannel( np.lib.pad( src_bytes[:,:,0], 1, 'constant', constant_values=0 ) )
        census_b = censusTransformSingleChannel( np.lib.pad( src_bytes[:,:,1], 1, 'constant', constant_values=0 ) )
        census_c = censusTransformSingleChannel( np.lib.pad( src_bytes[:,:,2], 1, 'constant', constant_values=0 ) )
        census = np.dstack( (census_a,census_b,census_c) )



    return census



##iname = 'Glasses0S.png'
#iname = 'lena.bmp'
#
#im = cv2.imread( iname, cv2.IMREAD_COLOR )
#print 'im.shape : ', im.shape, im.dtype
#
#startTime = time.time()
#census = censusTransform( im)
#print 'Elapsed time : ', time.time() - startTime
#print 'census.shape : ', census.shape, census.dtype
#
#cv2.imshow( 'org', im )
#cv2.imshow( 'census', census )
#
#cv2.waitKey(0)

##Get the source image
#src_img = Image.open(iname)
#src_img.show()
#
#w, h = src_img.size
#print('image size: %d x %d = %d' % (w, h, w * h))
#print('image mode:', src_img.mode)
#
##Convert image to Numpy array
#src_bytes = np.asarray(src_img)
#
#startTime = time.time()
#census = censusTransform(src_bytes)
#print 'time taken : ', time.time() - startTime
#
#
##Convert transformed data to image
#out_img = Image.fromarray(census)
#out_img.show()
#out_img.save(oname)
