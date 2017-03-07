""" Input 2 images. Compute vlad_word (and char) and report the similarity score"""

import cv2
import numpy as np
import time
import code


im1_raw = cv2.imread( 'other_seqs/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_000000.ppm' )
im2_raw = cv2.imread( 'other_seqs/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_000010.ppm' )

im_ = cv2.resize( im1_raw, (320,240) )
im1 = cv2.cvtColor( im_.astype('uint8'), cv2.COLOR_RGB2BGR )

im_ = cv2.resize( im2_raw, (320,240) )
im2 = cv2.cvtColor( im_.astype('uint8'), cv2.COLOR_RGB2BGR )

cv2.imshow( 'im1', im1 )
cv2.imshow( 'im2', im2 )

cv2.waitKey(0)
