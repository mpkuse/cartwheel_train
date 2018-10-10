""" Class for generating query, positive, negative sample set (16) from `keezi_walks`

    There are several city walking videos on youtube. I have downloaded several
    and made a dataset. This class will load up all the videos (around 15 videos of
    2 hrs each). This class defines a renderer similar in interface to TimeMachineRender.

    basically a function step() which return a tuple (q, nP, nN) ie. a query image.
    nP number of images similar to q and nN dis-similar images.

    In addition there will also be an function step_random() whih return n random images

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 15th Aug 2017.
"""

import numpy as np
#import matplotlib.pyplot as plt
import time
import cv2
import code
import math
import glob
import pickle

#
import TerminalColors
tcolor = TerminalColors.bcolors()


class WalksRendererOnline:
    def __init__( self, db_path ):
        self.db_path = db_path
        print tcolor.OKGREEN, 'WalksRenderer.db_path : ', db_path, tcolor.ENDC

        print tcolor.OKBLUE, 'Video Files : ', tcolor.ENDC
        self.all_files = []
        for _i, file_name in enumerate( glob.glob( db_path+"/*.mkv" ) +  glob.glob( db_path+"/*.mp4" )):
            print file_name
            self.all_files.append( file_name )

    def proc_vfile( self, vfilename ):
        # vfilename = 'Amsterdam.mkv'
        # vfilename = self.all_files[5]
        print 'vfilename = ', vfilename
        txt = np.loadtxt( vfilename+'.txt', delimiter=',', dtype='int32' )
        cap = cv2.VideoCapture( vfilename )
        assert( cap.isOpened() )
        nFrames = cap.get( cv2.CAP_PROP_FRAME_COUNT )

        L = [ [txt[0]] ]
        for i in range( txt.shape[0]-1 ):
            if abs(txt[i+1,0] - txt[i,0]) < 500:
                L[-1].append( txt[i+1] )
            else:
                L.append( [txt[i+1] ] )

        # code.interact( local=locals() )

        print 'nSegments = ', len(L)
        for l in L:
            l = np.array( l )
            # print l
            print 'len_of_this_seg=', len(l)


            cap.set( cv2.CAP_PROP_POS_FRAMES, l[0,0] )
            ret, frame0 = cap.read()
            IM0 = cv2.resize( cv2.blur(frame0, (5,5)), (320,240) )#, fx=0.2, fy=0.2 )


            cap.set( cv2.CAP_PROP_POS_FRAMES, l[0,1] )
            ret, frame1 = cap.read()
            IM1 = cv2.resize( cv2.blur(frame1, (5,5)), (320,240) )#, fx=0.2, fy=0.2 )


            cv2.imshow( 'frame0', IM0 )
            cv2.imshow( 'frame1', IM1 )
            cv2.waitKey(0)




    def proc(self):
        c = self.proc_vfile( self.all_files[-4] )






class WalksRenderer:
    def __init__( self, db_path ):
        self.db_path = db_path
        print tcolor.OKGREEN, 'WalksRenderer.db_path : ', db_path, tcolor.ENDC

        self.captures = []

        print tcolor.OKBLUE, 'Video Files : ', tcolor.ENDC
        for _i, file_name in enumerate( glob.glob( db_path+"/*.mkv" ) ):
        # for _i, file_name in enumerate( ['Valparasio_Chile.mkv', 'Tokyo.mkv'] ):
            cap = cv2.VideoCapture( file_name )

            if cap.isOpened():

                nFrames = cap.get( cv2.CAP_PROP_FRAME_COUNT )
                print tcolor.OKBLUE, '+    %03d nFrames=%06d' %(_i, nFrames), file_name, tcolor.ENDC
                self.captures.append( (cap, nFrames) )
            else:
                print tcolor.FAIL, '~    %03d' %(_i), file_name, tcolor.ENDC


    # Note that every image defined by (capture_id, frame_id)
    def _query( self, n ):

        to_ret = []
        for i in range(n):
            capture_id = np.random.randint( low=0, high=len(self.captures) ) # select any of the capture
            frame_id = np.random.randint( low=0 , high=self.captures[ capture_id ][1] ) #for a given capture select any frame
            to_ret.append( (capture_id, frame_id) )

        return to_ret


    def _similar_to( self, nP,capture_id, frame_id  ):
        to_ret = []
        for i in range( nP ):
            # r = np.floor( np.random.normal( loc=capture_id, scale=500 ) )
            r = np.random.randint( low=frame_id-200, high=frame_id+200 )

            # bounding clipping
            r = max( 0, int(r) )
            r = min( r, self.captures[capture_id][1])

            to_ret.append( (capture_id, r ) )
        return to_ret



    # pos_list = [ (capture_id, frame_id),  (capture_id, frame_id),   (capture_id, frame_id) ...  ]
    def _load_images( self, pos_list, apply_distortions=False ):
        images = []
        for capture_id, frame_id in pos_list:
            # print 'load', capture_id, frame_id
            self.captures[capture_id][0].set( cv2.CAP_PROP_POS_FRAMES, frame_id )
            ret, frame = self.captures[capture_id][0].read()
            IM = cv2.resize( cv2.blur(frame, (5,5)), (320,240) )

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
                # print 'irot: ', irot
                image_rotated = rotate_image(IM, irot)
                image_rotated_cropped = crop_around_center(
                    image_rotated,
                    *largest_rotated_rect(
                        image_width,
                        image_height,
                        math.radians(irot)
                    ))
                IM = cv2.resize( image_rotated_cropped, (320,240) )


            images.append( IM )
        return images #np.concatenate( images, axis=1)


    # Return nP number of positive samples, nN number of negative samples
    def step( self, nP, nN, return_gray=False):
        #TODO : Consider using return_gray when loading images

        _q = self._query( 1 )


        _sims = self._similar_to( nP, _q[0][0], _q[0][1] )
        _different = self._query( nN )



        startLoad = time.time()
        images_q    = self._load_images( _q )
        images_sim  = self._load_images(_sims, apply_distortions=True)
        images_diff = self._load_images(_different)

        # print 'Took %4.2fms to load images' %(1000. * (time.time() - startLoad) )
        cv2.imshow( 'images_q', np.concatenate(images_q, axis=1) )
        cv2.imshow( 'images_sim', np.concatenate(images_sim, axis=1) )
        cv2.imshow( 'images_diff', np.concatenate(images_diff, axis=1) )

        cv2.moveWindow( 'images_sim', 0, 300 )
        cv2.moveWindow( 'images_diff', 0, 600 )
        cv2.waitKey(1)

        return np.concatenate( (images_q, images_sim, images_diff  ), axis=0 ).astype('float32'), np.zeros( (16,4) )




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
