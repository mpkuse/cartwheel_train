""" Class for generating query, positive, negative sample set (16) from `Tokyo_TM`

    Makes use of Tokyo_TM data from original NetVLAD paper. This class provides
    an interface similar to PandanRender.NetVLADrenderer. Can be used to generate
    a training sample with 16 images. 1st image being query. Next nP being
    postive sample (ie. same place as query). Next nN being negative samples.

    The Tokyo_TM data set was obtained from Akihiko_Torii. Paper to reference
    NetVLAD : CNN architecture for weakly supervised place recognition

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 22nd June, 2017
"""

import scipy.io #for reading .mat files
import numpy as np
#import matplotlib.pyplot as plt
# import pyqtgraph as pg
import time
import cv2
import code
import math

#
import TerminalColors
tcolor = TerminalColors.bcolors()

class TimeMachineRender:
    ## Give base path of Tokyo_TM. eg: TTM_BASE = 'data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/'
    ## nP (number of positive samples)
    ## nN (number of negative samples)
    def __init__( self, TTM_BASE, nP=5, nN=10 ):
        print tcolor.HEADER, 'TimeMachineRender : TokyoTM', tcolor.ENDC
        print 'TTM_BASE : ', TTM_BASE
        self.TTM_BASE = TTM_BASE

        #
        # Load the .mat file containing list of images, location data, time
        print 'Opening File : ', TTM_BASE+'/tokyoTM_train.mat'
        mat = scipy.io.loadmat( TTM_BASE+'/tokyoTM_train.mat' )
        dbStruct = mat['dbStruct']
        utmQ = dbStruct['utmQ'].item()
        utmDb = dbStruct['utmDb'].item() #2x...

        dbImageFns = dbStruct['dbImageFns'].item()
        qImageFns = dbStruct['qImageFns'].item()
        #can do dbImageFns[0][0][0], dbImageFns[1][0][0], dbImageFns[2][0][0], dbImageFns[3][0][0] , ...

        dbTimeStamp = dbStruct['dbTimeStamp'].item()[0,:]
        qTimeStamp = dbStruct['qTimeStamp'].item()[0,:]


        #
        # Process this data and make a hierarchy
        # root
        #   | - location_i
        #   |       | - timeStamp (eg. 200907 etc)
        #   .       |       | - image-path-1
        #           |       | - image-path-2
        pyDB = {}
        for i in range( utmQ.shape[1] ):
            # print 'Db', utmDb[0,i], utmDb[1,i]#, dbTimeStamp[i]
            _x = utmDb[0,i] #float32
            _y = utmDb[1,i] #float32
            _t = dbTimeStamp[i] #int
            _file_name = dbImageFns[i][0][0] #string

            if ( str(_x), str(_y) ) in pyDB.keys():
        	if _t in pyDB[ str(_x), str(_y) ].keys():
                    pyDB[ str(_x), str(_y) ][_t].append( _file_name )
        	else:
                    pyDB[ str(_x), str(_y) ][_t] = []
                    pyDB[ str(_x), str(_y) ][_t].append( _file_name )
            else:
                pyDB[ str(_x), str(_y) ] = {}
                pyDB[ str(_x), str(_y) ][_t] = []
                pyDB[ str(_x), str(_y) ][_t].append( _file_name  )

        print tcolor.OKGREEN, 'Database contains ', len(pyDB.keys()), ' uniq locations', tcolor.ENDC
        self.pyDB = pyDB

    ## imshow() images of a location across various timestamps. Modify as need be to lookinto the data
    def debug_display_image_samples(self):
        pyDB = self.pyDB
        locs = pyDB.keys()
        for l in locs:
        	yrs_list = pyDB[l].keys()
        	print 'loc=',l, len(yrs_list), yrs_list

        	if len(yrs_list) < 2:
        	    continue

        	win_list = []
        	for yi,y in enumerate(yrs_list):
        		print '    ', y, len( pyDB[l][y] )
        		#for f in pyDB[l][y]:
        		#	print '        ', f
        		for circ_i, circ in enumerate([-2,-1,0,1,2]):
        		   	cv2.namedWindow( str(y)+'_'+str(circ), cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE )

        			cv2.imshow( str(y)+'_'+str(circ), cv2.resize(cv2.imread( self.TTM_BASE+'/images/'+pyDB[l][y][circ] ), (0,0), fx=0.25, fy=0.25 ) )
        			cv2.moveWindow( str(y)+'_'+str(circ), 180*yi, 10+180*circ_i )
        			print 200*yi, 10+200*circ_i

        			win_list.append( str(y)+'_'+str(circ) )

        	cv2.waitKey(0)
        	for w in win_list:
        		cv2.destroyWindow(w)


    # Generate a query image randomly. returns loc_idx, yr_idx, im_idx
    def _query(self, exclude_loc_idx=None):
        pyDB = self.pyDB
        randint = np.random.randint # randint( 10 ) #will give an integer between 0 and 10

        # Pick a random location
        locs = pyDB.keys()
        # print 'exclude_loc_idx', exclude_loc_idx
        if exclude_loc_idx is None:
            q_li = randint( len(locs) ) #location_index
        else:
            q_li = randint( len(locs) ) #location_index
            while q_li == exclude_loc_idx:
                # print 'regenerate'
                q_li = randint( len(locs) ) #location_index

        # Given a location pick year
        yr_list = pyDB[ locs[q_li] ].keys()
        q_yr = randint( len(yr_list) ) #year_index

        # Given a location, given a year, pick an image
        im_list = pyDB[ locs[q_li] ][yr_list[q_yr]]
        q_i  = randint( len(im_list) )

        loc_idx = q_li
        yr_idx = q_yr
        im_idx = q_i
        return (loc_idx, yr_idx, im_idx)




    # Generate `n` query similar to (loc_idx, yr_idx, im_idx). n need to be 2 or more
    # How it works :
    #   a) ``loc_idx, yr_idx, im_indx +- 1``  --> 2
    #   b) generate 2*n ``loc_idx, choose(yr_idx), im_idx + choose(-1,0,1)``
    #   c) choose n-2 out of these 2*n
    def _similar( self, n, loc_idx, yr_idx, im_idx ):
        pyDB = self.pyDB
        randint = np.random.randint # randint( 10 ) #will give an integer between 0 and 10

        # print '_similar()'
        # print 'inp', loc_idx, yr_idx, im_idx

        loc = pyDB.keys()[loc_idx]
        yr_list = pyDB[ loc ].keys()
        yr = yr_list[ yr_idx ]
        im_list = pyDB[loc][yr]

        # im_idx+1, im_idx-1 x 2
        A = []
        r_p = ( loc_idx, yr_idx, ( (im_idx+1) % len(im_list)) )
        r_n = ( loc_idx, yr_idx, ( (im_idx-1) % len(im_list)) )

        A.append( r_p )
        A.append( r_n )
        # print 'ret', r_p
        # print 'ret', r_n


        # if `yr_list` has 3 or more
        S1 = []
        # print 'len(yr_list)', len(yr_list)
        if len(yr_list) >= 3:
            for i in range(n-2):
                __im_indx = im_idx+int(np.random.randn()/2.)
                S1.append( (loc_idx, randint( len(yr_list) ), __im_indx%len(im_list) ) )
            return A+S1




        # choose another year (if possible)
        # Generate n+5 and then choose n-2 out of these
        B = []
        for i in range(n*2):
            if len(yr_list) == 1:
                r = randint( -2, 3) #if only 1 year data, den, -2, -1, 0, 1, 2
            else:
                r = randint( -1, 2 ) #returns either -1 or 1 or 0

            g = loc_idx, randint(len(yr_list)), ( (im_idx+r) % len(im_list))
            B.append( g )
            # print 'ret', g

        # Choose n-2 from B
        import random
        if n>2:
            C = random.sample( B, n-2 )
        else :
            C = []


        return A + C



    ## Generate `n` number of images different than the given one
    def _different(self, n, loc_idx, yr_idx, im_idx ):
        A = []
        for i in range( n ):
            A.append( self._query( exclude_loc_idx=loc_idx ) )
        return A

    ## Given a set for example, L = [(25, 0, 7), (25, 0, 5), (25, 0, 6), (25, 0, 4), (25, 0, 4)]
    ## Load corresponding images. Returns a np.array of size nx240x320x3
    ## If apply_distortions is true, random distortions will be applied. Currently planar rotations with angles as Gaussian distribution centered at 0, sigma=25
    def _get_images(self, L, resize=None, apply_distortions=False, return_gray=False, PRINTING=False):
        pyDB = self.pyDB
        A = []
        for loc_idx, yr_idx, im_idx in L:
            loc = pyDB.keys()[loc_idx]
            yr_list = pyDB[ loc ].keys()
            yr = yr_list[ yr_idx ]
            im_list = pyDB[loc][yr]
            try:
                im_name = im_list[ im_idx ]

                # print loc_idx, yr_idx, im_idx
                file_name = self.TTM_BASE+'/images/'+im_name
                if PRINTING:
                    print 'imread : ', file_name
                # TODO blur before resizing
                if resize is None:
                    IM = cv2.imread( file_name )
                else:
                    IM = cv2.resize( cv2.imread( file_name ) , resize  )
                # IM = cv2.resize( cv2.imread( file_name ) , (160,120)  )
            except:
                print 'im_indx error', im_list
                IM = np.zeros( (240, 320, 3) ).astype('uint8')

            # Random Distortion
            if apply_distortions == True and np.random.rand() > 0.5: #apply random distortions to only 50% of samples
                #TODO: Make use of RandomDistortions class (end of this file) for complicated Distortions, for now quick and dirty way
                # # Planar rotate IM, this rotation gives black-borders, need to crop
                # rows,cols, _ = IM.shape
                # irot = np.random.uniform(-180,180 )#np.random.randn() * 25.
                # M = cv2.getRotationMatrix2D((cols*.5,rows*.5),irot,1.)
                # dst = cv2.warpAffine(IM,M,(cols,rows))
                # IM = dst

                # Planar rotation, cropped. adopted from `test_rot-test.py`
                image_height, image_width = IM.shape[0:2]
                image_orig = np.copy(IM)
                irot = np.random.uniform(-180,180 )#np.random.randn() * 25.
                image_rotated = rotate_image(IM, irot)
                image_rotated_cropped = crop_around_center(
                    image_rotated,
                    *largest_rotated_rect(
                        image_width,
                        image_height,
                        math.radians(irot)
                    ))
                IM = cv2.resize( image_rotated_cropped, (320,240) )




            if return_gray == True:
                IM_gray = cv2.cvtColor( IM, cv2.COLOR_BGR2GRAY )
                IM = np.expand_dims( IM_gray, axis=2 )


            # A.append( IM[:,:,::-1] )
            A.append( IM )

        return np.array(A)
        #cv2.imshow( 'win', np.concatenate( A, axis=1 )  )
        #cv2.waitKey(0)




    # Gives out `nP` number of positive samples of query image. `nN` number of negative samples.
    # Note, query image is the 0th image. Next nP will be positive, next nN will be negative.
    # return_gray=True will return a (N,240,320,1), ie gray scale images
    def step(self, nP, nN, resize=None, apply_distortions=False, return_gray=False, ENABLE_IMSHOW=False):
        # np.random.seed(1)
        # Will generate a total of 1+nP+nN number of images. 1st is the query image (choosen randomly)
        # Next nP will be positive Samples. Next nN will be negative samples

        loc_idx, yr_idx, im_idx = self._query()
        sims = self._similar( nP, loc_idx, yr_idx, im_idx )
        diffs = self._different(nN, loc_idx, yr_idx, im_idx)

        # print '---'
        # print 'q : ', loc_idx, yr_idx, im_idx
        # print 's : ', sims
        # print 'd : ', diffs
        
        PRINTING = False
        q_im = self._get_images( [(loc_idx, yr_idx, im_idx)], resize=resize, apply_distortions=apply_distortions, return_gray=return_gray , PRINTING=PRINTING )
        sims_im = self._get_images(sims[0:nP], resize=resize, apply_distortions=apply_distortions, return_gray=return_gray, PRINTING=PRINTING)
        diffs_im = self._get_images(diffs, resize=resize, apply_distortions=apply_distortions, return_gray=return_gray, PRINTING=PRINTING)


        # print q_im.shape
        # print sims_im.shape
        # print diffs_im.shape
        if ENABLE_IMSHOW:

            cv2.imshow( 'q_im', np.concatenate( q_im, axis=1)[:,:,::-1] )
            cv2.imshow( 'sims_im', np.concatenate( sims_im, axis=1)[:,:,::-1] )
            cv2.imshow( 'diffs_im', np.concatenate( diffs_im, axis=1)[:,:,::-1] )
            cv2.waitKey(1)

        return np.concatenate( (q_im, sims_im, diffs_im), axis=0 ).astype('float32'), np.zeros( (16,4) )

    # Gives out all 16 totally random
    def step_random( self, count ):
        loc_idx, yr_idx, im_idx = self._query()
        diffs = self._different(count, loc_idx, yr_idx, im_idx)
        diffs_im = self._get_images(diffs)

        # cv2.imshow( 'diffs_im', np.concatenate( diffs_im, axis=1)[:,:,::-1] )
        # cv2.waitKey(0)

        return diffs_im.astype('float32'), np.zeros( (count,4) )


#TODO: Later as you want to try out more complicated distortions, write this class
class RandomDistortions:
    def __init__(self):
        #$ TODO: Try and get these passed as constructor params

        # Rotation Parameters
        self.M = 0

    def distort_image(self, IM ):
        return None



# Rotation (borderless)
def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]
