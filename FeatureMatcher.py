"""
    Takes the rendered images as input. This consists on 1 query image
    n positive samples and m positive sample. Gives out point-feature matching
    between I_q and each of the positive sets, I_{P_i}

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 21st Mar, 2018
"""

import numpy as np
import code
import sys
import os
import time
import cv2
cv2.ocl.setUseOpenCL(False)

from gms_matcher import GmsMatcher, DrawingType

try:
    from DaisyMeld.daisymeld import DaisyMeld
except:
    print 'If you get this error, your DaisyMeld wrapper is not properly setup. You need to set DaisyMeld in LD_LIBRARY_PATH. and PYTHONPATH contains parent of DaisyMeld'
    print 'See also : https://github.com/mpkuse/daisy_py_wrapper'
    print 'Do: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s/DaisyMeld' %(os.getcwd())
    print 'do: export PYTHONPATH=$PYTHONPATH:$HOME/catkin_ws/src/nap/scripts'
    quit()

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


class MatcherDaisy:
    def __init__(self):
        self.dai1 = DaisyMeld(240, 320, 0)
        self.dai2 = DaisyMeld(240, 320, 0)


        # FLANN Matcher
        # Prepare FLANN Matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)


    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint(self):
        sys.stdout = sys.__stdout__

    def _get_daisy( self, im, obj_id ):
        # Need to give gray image input, float32
        if obj_id == 1:
            self.dai1.do_daisy_computation( im )
            vi = self.dai1.get_daisy_view()
            return vi

        if obj_id == 2:
            self.dai2.do_daisy_computation( im )
            vi = self.dai2.get_daisy_view()
            return vi

        print 'ERROR in MatcherDaisy._get_daisy(). obj_id can be either 1 or 2'
        quit()

    def _kmeans_labels_to_image( self, im, center, label  ):
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((im.shape))
        return res2

    def _kmeans( self, im, im_array, K=8 ):
        """ im: mxnx3 ; im_array : Nxmxnx3 (an array of N images)

            returns 2 entities. 1st is corresponding to im. 2nd
            corresponding to im_array. The entity has kmeans
            residue (scalar), labels and centers.

        """

        assert( len(im.shape) == 3, "im should be mxnx3" )
        assert( len(im_array.shape) == 4, "im should be Nxmxnx3" )

        # CSV Space
        # hsv_im = cv2.cvtColor( im, cv2.COLOR_BGR2HSV )
        # hsv_im_array = []
        # for i in range( im_array.shape[0] ):
        #     hsv_imx = cv2.cvtColor( im_array[0,:,:,:], cv2.COLOR_BGR2HSV )
        #     hsv_im_array.append( hsv_imx )
        # hsv_im_array = np.array( hsv_im_array )
        # code.interact( local=locals() )


        Z1 = np.float32( im.reshape( (-1,im.shape[2]) ) )

        Z2 = []
        for i in range( im_array.shape[0] ):
            n = im_array.shape[3]
            Z2.append(  np.float32( im_array[i,:,:,:].reshape( (-1,n) ) ) )

        # Kmeans params
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


        # Kmeans for image1
        ret1,label1,center1=cv2.kmeans(Z1,K,None,criteria,10,cv2.KMEANS_PP_CENTERS 	)
        Lx = np.copy(label1)

        # K means on others using centers from image1
        kmeans_im_array = []
        for i in range( im_array.shape[0] ):
            ret2,label2,center2=cv2.kmeans(Z2[i],K,label1,criteria,1,cv2.KMEANS_USE_INITIAL_LABELS)
            # save ret2, label2, center2 for all the images
            kmeans_im_array.append( (ret2,label2.flatten(),center2)  )

        return (ret1,Lx.flatten(),center1), kmeans_im_array


    def s_overlay( self, im, mask ):
        im[:,:,2] = mask.astype( 'uint8' )
        return im

    def match_me_simple(self, im1, im2 ):
        im1 = im1.astype('uint8')
        im2 = im2.astype('uint8')

        im1_gray = cv2.cvtColor( im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor( im2, cv2.COLOR_BGR2GRAY)


        # Compute daisy descriptors for im1 and im2
        vi_1 = self._get_daisy( im1_gray.copy().astype( 'float32' ), 1 ) #240, 320, 20
        vi_2 = self._get_daisy( im2_gray.copy().astype( 'float32' ), 2 )
        print vi_1.shape , vi_2.shape



        # Nearest neighbour
        QR = np.tensordot( vi_1, vi_2, (2,2) ) #240x320x240x320
        for r in range(240):
            for c in range( 320 ):
                M = QR[r,c,:,:]
                code.interact( local=locals() )







    def match_me(self, im1, im2 ):
        im1 = im1.astype('uint8')
        im2 = im2.astype('uint8')

        im1_gray = cv2.cvtColor( im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor( im2, cv2.COLOR_BGR2GRAY)



        # Compute daisy descriptors for im1 and im2
        vi_1 = self._get_daisy( im1_gray.copy().astype( 'float32' ), 1 ) #240, 320, 20
        vi_2 = self._get_daisy( im2_gray.copy().astype( 'float32' ), 2 )
        print vi_1.shape , vi_2.shape

        # cv2.imshow( 'vi_1', (255*vi_1[:,:,0]).astype('uint8') )
        # cv2.imshow( 'vi_2', (255*vi_2[:,:,0]).astype('uint8') )

        cv2.imshow( 'im1', im1 )
        cv2.imshow( 'im2', im2 )


        # Kmeans as a proxy for semantic segmentation for domain reduction for daisy matching
        K = 20
        kmeans_im1, kmeans_imarray = self._kmeans( im1, np.array( [im2] ), K=K )
        # kmeans_im1, kmeans_imarray = self._kmeans( vi_1, np.array( [vi_2] ), K=K )


        # Simple nearest neighbour matching
        #   for each pixel find 2 nearest neighbours and do Lowe's style elimination
        for k in range(K): #loop over each clusters
            mask = ( np.reshape(kmeans_im1[1], im1.shape[0:2] ) == k ).astype('uint8') *255

            mask2 = ( np.reshape(kmeans_imarray[0][1], im2.shape[0:2] ) == k ).astype('uint8') *255
            print k
            cv2.imshow( 'K1', self.s_overlay( im1, mask ) )
            cv2.imshow( 'K2', self.s_overlay( im2, mask2 ) )
            # code.interact( local=locals(), banner="loop thru each cluster " )
            cv2.waitKey(0)





        # Remove more matches as per f-test


        # Draw matches

        cv2.imshow( 'im1', im1.astype('uint8') )
        cv2.imshow( 'im2', im2.astype('uint8') )
