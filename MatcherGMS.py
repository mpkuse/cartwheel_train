"""
    Takes the rendered images as input. This consists on 1 query image
    n positive samples and m positive sample. Gives out point-feature matching
    between I_q and each of the positive sets, I_{P_i}

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 21st Mar, 2018
"""

import numpy as np
import code
import cv2

from gms_matcher import GmsMatcher, DrawingType
cv2.ocl.setUseOpenCL(False)


class MatcherGMS:
    def __init__(self):
        orb = cv2.ORB_create(10000)
        orb.setFastThreshold(0)

        surf = cv2.xfeatures2d.SURF_create(4000)
        bf = cv2.BFMatcher()

        if cv2.__version__.startswith('3'):
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        # self.gms = GmsMatcher(orb, matcher)
        self.gms = GmsMatcher(surf, bf)


    # Input:
    # One query image I_q (axbx3)
    # a positive set with say 10 images (10xaxbx3)
    def match( self, I_q, P ):

        for i in range(P.shape[0]):
            im1 = cv2.cvtColor( I_q.astype('uint8'), cv2.COLOR_RGB2GRAY )
            im2 = cv2.cvtColor( P[i,:,:,:].astype('uint8'), cv2.COLOR_RGB2GRAY )

            im1_c = cv2.cvtColor( I_q.astype('uint8'), cv2.COLOR_RGB2BGR )
            im2_c = cv2.cvtColor( P[i,:,:,:].astype('uint8'), cv2.COLOR_RGB2BGR )

            cv2.imshow( 'im1', im1 )
            cv2.imshow( 'im2', im2 )
            # cv2.waitKey(0)

            matches = self.gms.compute_matches( im1, im2 )
            # code.interact( local=locals() )
            self.gms.draw_matches( im1_c, im2_c, DrawingType.ONLY_LINES)
            # code.interact( local=locals() )
