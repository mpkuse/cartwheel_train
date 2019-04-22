""" Class for generating query, positive and negative sample set for
    Pitssburg street view data set.

    The dataset is bigger than tokyo time machine dataset. About 10K uniq locations.

    The dataset contains of several folders 000, 001, ..., 010.
    Each of these folders contain 1000 locations. Each location has 24 images.
    {pitch1, pitch2} x {yaw1, yaw2,...,yaw12}. File names like :
    {folder_name}_{pitchi}_{yawj}.jpg

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 12th Dec, 2017
"""


# import scipy.io #for reading .mat files
import numpy as np
#import matplotlib.pyplot as plt
# import pyqtgraph as pg
import time
import cv2
import code
import math
import os
import glob
import code
import copy
import random
import glob
#
import TerminalColors
tcolor = TerminalColors.bcolors()

class PittsburgRenderer:
    def __init__( self, PTS_BASE ):
        print tcolor.OKGREEN, '--------------PittsburgRenderer::__init__--------------', tcolor.ENDC
        self.PTS_BASE = PTS_BASE
        print 'PTS_BASE:', PTS_BASE

        self.folder_list = []

        # See if this looks like the correct folder,
        # A: if folders by name 000, 001, 002, ..., 010 exist
        for i in range(100):
            folder_name = '%s/%03d' %(self.PTS_BASE, i)
            # if i==10:
                # print 'ignore 5 because the folder is corrupt. this is manual intervention'
                # continue
            if os.path.isdir( folder_name ):
                print 'Look for folder : ', folder_name, '\t',
                print 'exists'


                # look inside this folder to see how many images
                if True:
                    nk = 0
                    for k in range(1000):
                        # print 'check : ' , folder_name+'/%3d%3d_pitch1_yaw1.jpg' %(i,k)
                        k_decision = os.path.isfile( folder_name+'/%03d%03d_pitch1_yaw1.jpg' %(i,k) )
                        if k_decision:
                            nk+=1

                    print '\ttotal_images=', len(glob.glob( folder_name+'/*.jpg' )),
                    print 'total_uniq_locations=', nk

                if nk < 999:
                    continue
                self.folder_list.append(i)

            else:
                pass
                #print 'does not exist, break'
                #break;
            # print 'Check if folder \'%s\' exist' %(folder_name)

        if len(self.folder_list) == 0:
            print tcolor.FAIL, 'Doesnot look like streetview pitsburg db', tcolor.ENDC
            quit()
        print tcolor.OKGREEN, 'END--------------PittsburgRenderer::__init__--------------', tcolor.ENDC

    def _tuple_to_filename( self, t ):
        filename = '%s/%03d/%03d%03d_pitch%d_yaw%d.jpg' %(self.PTS_BASE, t[0], t[0], t[1], t[2], t[3])
        return filename




    def _query( self, exclude=[-1,-1,-1,-1]):
        """ Generate a 4-tuple. (folderId, imageId, pitchId, yawId)
                folderId : 000, 001, ..., 010
                imageId  : 000, 001, ..., 999
                pitchid  : 1, 2
                yawid    : 1, 2, ..., 12

                exclude is also a 4-tuple
        """
        assert len(exclude)==4, "Length of exclude is not 4"

        if self.folder_list is None:
            folderId = np.random.randint( 10 ) #folder 010 contains only 585 images, so ignoring it.
            while folderId == exclude[0]:
                folderId = np.random.randint( 10 )
        else:
            folderId = random.choice( self.folder_list )

        imageId  = np.random.randint( 1000 )
        while imageId == exclude[1]:
            imageId  = np.random.randint( 1000 )

        pitchId  = np.random.randint( 2 ) + 1
        while pitchId == exclude[2]:
            pitchId  = np.random.randint( 2 ) + 1

        yawId    = np.random.randint( 12 ) + 1
        while yawId == exclude[3]:
            yawId    = np.random.randint( 12 ) + 1

        return [folderId, imageId, pitchId, yawId]

    def _similar_to( self, n, t ):
        """ Gives a tuple similar to specified tuple """

        if t[2] == 1:
            _t2 = 2
        else:
            _t2 = 1

        L = []

        L.append( [t[0], t[1], _t2, t[3] ] )

        #only change pitch and yaw, ie. t[2], t[3]
        new_pitch = np.random.randint( 2, size=n ) + 1
        new_yaw = np.random.randn(n)*1. + t[3]
        # new_yaw = int( np.random.randn(n) * 8 + t[3] )



        for i in range(1, n):
            out = copy.copy(t)

            out[2] = new_pitch[i]
            out[3] = int(np.floor(new_yaw[i])) % 12 + 1
            L.append( out )

        return L


    def _different_than( self, n, t ):
        L = []
        for i in range(n):
            L.append( self._query( exclude=t ) )

        return L

    def _get_images( self, L, apply_distortions, return_gray, resize ):
        A = []
        for l in L:
            # print 'l=',l
            fname = self._tuple_to_filename( l )
            # print 'Load Image', fname
            try:
                if resize is None:
                    IM = cv2.imread( fname )
                else:
                    assert(len(resize) == 2)
                    IM = cv2.resize( cv2.imread( fname ) , resize  )
            except:
                IM = np.zeros( (resize[0], resize[1], 3) ).astype('uint8')


            if return_gray == True:
                IM_gray = cv2.cvtColor( IM, cv2.COLOR_BGR2GRAY )
                IM = np.expand_dims( IM_gray, axis=2 )




            # A.append( IM[:,:,::-1] ) # return rgb
            A.append( IM[:,:,:] )

        return np.array(A)

    def _add_image_caption( self, im, txt ) :

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


    def step( self, nP, nN, apply_distortions=False, return_gray=False, resize=None, ENABLE_IMSHOW=False ):
        """
        Will return 2 matrix.
        a: size=(1+nP+nN)xresize[0]xresize[1]x3. example. (1+10+5)x320x240x3
        b: size=(1+nP+nN)x4. These are just zeros here, they are here for compatibility. but it serves no purpose currently.
        """

        # return self.preload_step( nP, nN, apply_distortions, return_gray, ENABLE_IMSHOW )
        q_tup = self._query()
        sim_tup = self._similar_to( nP, q_tup)
        dif_tup = self._different_than( nN, q_tup )

        # print 'q_tup=  ' , q_tup
        # print 'sim_tup=' , sim_tup
        # print 'dif_tup=' , dif_tup

        q_im = self._get_images( [q_tup], apply_distortions=apply_distortions, return_gray=return_gray, resize=resize )
        sim_im = self._get_images( sim_tup, apply_distortions=apply_distortions, return_gray=return_gray, resize=resize )
        dif_im = self._get_images( dif_tup, apply_distortions=apply_distortions, return_gray=return_gray, resize=resize )
        # code.interact( local=locals() )

        # print 'q_im.shape=  ' , q_im.shape
        # print 'sim_im.shape=' , sim_im.shape
        # print 'dif_im.shape=' , dif_im.shape


        if ENABLE_IMSHOW:
            # rgb <---> bgr
            # cv2.imshow( 'q_im', np.concatenate( q_im, axis=1)[:,:,::-1] )
            # cv2.imshow( 'sims_im', np.concatenate( sim_im, axis=1)[:,:,::-1] )
            # cv2.imshow( 'diffs_im', np.concatenate( dif_im, axis=1)[:,:,::-1] )

            # Put caption for q_im
            xstr = 'folderID=%d;imageID=%d;pitchID=%d;yawID=%d' %(q_tup[0], q_tup[1], q_tup[2], q_tup[3] )
            cv2.imshow( 'q_im_show', self._add_image_caption( q_im[0].astype('uint8'), xstr ) )

            # put caption for sim_im
            sim_im_show = []
            for z in range( len(sim_im) ):
                if z == 0:
                    xstr = 'folderID=%d;imageID=%d;pitchID=%d;yawID=%d' %( (sim_tup[z])[0], (sim_tup[z])[1], (sim_tup[z])[2], (sim_tup[z])[3] )
                else:
                    xstr = '  %d;  %d;  %d;  %d' %( (sim_tup[z])[0], (sim_tup[z])[1], (sim_tup[z])[2], (sim_tup[z])[3] )
                sim_im_show.append( self._add_image_caption( sim_im[z].astype('uint8'), xstr ) )
            cv2.imshow( 'sim_im_show', np.concatenate( sim_im_show, axis=1)  )


            # put caption for dif_im
            diff_im_show = []
            for z in range( len(dif_im) ):
                if z == 0:
                    xstr = 'folderID=%d;imageID=%d;pitchID=%d;yawID=%d' %( (dif_tup[z])[0], (dif_tup[z])[1], (dif_tup[z])[2], (dif_tup[z])[3] )
                else:
                    xstr = '  %d;  %d;  %d;  %d' %( (dif_tup[z])[0], (dif_tup[z])[1], (dif_tup[z])[2], (dif_tup[z])[3] )
                diff_im_show.append( self._add_image_caption( dif_im[z].astype('uint8'), xstr ) )
            cv2.imshow( 'diff_im_show', np.concatenate( diff_im_show, axis=1)  )



            # cv2.imshow( 'q_im', np.concatenate( q_im, axis=1)       )
            # cv2.imshow( 'sims_im', np.concatenate( sim_im, axis=1)  )
            # cv2.imshow( 'diffs_im', np.concatenate( dif_im, axis=1) )
            cv2.waitKey(5)



        return np.concatenate( (q_im, sim_im, dif_im), axis=0 ).astype('float32'), np.zeros( (1+nP+nN,4) )

    def step_n_times( self, n_samples, nP, nN, return_gray=False, resize=(320,240), ENABLE_IMSHOW=False  ):
        print tcolor.OKGREEN, 'in fucntion PittsburgRenderer::step_n_times()', tcolor.ENDC
        D=[]
        for s in range(n_samples):
            a,_ = self.step(nP=nP, nN=nN, return_gray=return_gray, resize=(320,240), apply_distortions=False, ENABLE_IMSHOW=ENABLE_IMSHOW )
            if s%100 == 0:
                print tcolor.OKBLUE, 'get a sample from PTS_BASE=%s #%d of %d\t' %(self.PTS_BASE, s, n_samples),
                print a.shape, tcolor.ENDC

            D.append( a )
        if ENABLE_IMSHOW==True:
            print 'cv2.destroyAllWindows()'
            cv2.destroyAllWindows()

        return D

    def preload_step( self,  nP, nN, apply_distortions=True, return_gray=False, ENABLE_IMSHOW=False ):
        q_tup = self._query()
        sim_tup = self._similar_to( nP, q_tup)
        dif_tup = self._different_than( nN, q_tup )


        q_im = self._preload_get_images( [q_tup], apply_distortions=apply_distortions, return_gray=return_gray )
        sim_im = self._preload_get_images( sim_tup, apply_distortions=apply_distortions, return_gray=return_gray )
        dif_im = self._preload_get_images( dif_tup, apply_distortions=apply_distortions, return_gray=return_gray )


        if ENABLE_IMSHOW:
            cv2.imshow( 'q_im', np.concatenate( q_im, axis=1)[:,:,::-1] )
            cv2.imshow( 'sims_im', np.concatenate( sim_im, axis=1)[:,:,::-1] )
            cv2.imshow( 'diffs_im', np.concatenate( dif_im, axis=1)[:,:,::-1] )
            cv2.waitKey(5)



        return np.concatenate( (q_im, sim_im, dif_im), axis=0 ).astype('float32'), np.zeros( (1+nP+nN,4) )


    def _preload_get_images( self, L, apply_distortions, return_gray ):
        A = []
        for l in L:
            # print l
            fname = self._tuple_to_filename( l )

            # Find fname in array self.preload_fnames.
            x__indx = self.preload_fnames.index( fname )
            IM = self.preload_buffer[x__indx]

            # code.interact( local=locals() )

            # Apply Distortions
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




            A.append( IM[:,:,::-1] )

        return np.array(A)

    def preload_all_images( self, folder_list ):
        """ Loads all the Images into RAM """
        pass
        self.preload_buffer = []
        self.preload_fnames = []
        self.folder_list = folder_list

        # folderId : 000, 001, ..., 010
        # imageId  : 000, 001, ..., 999
        # pitchid  : 1, 2
        # yawid    : 1, 2, ..., 12
        estimated = len(folder_list)*1000*2*12
        cc = 0
        for folderId in folder_list:#range(0,11):
            for imageId in range(0,1000):
                for pitchid in [1,2]:
                    for yawid in range( 1, 13 ):
                        filename = self._tuple_to_filename( [folderId, imageId, pitchid, yawid] )

                        # Check if file exist
                        if os.path.isfile( filename ) == False:
                            continue

                        cc += 1
                        print '%d of %d: Read File' %(cc,estimated), filename
                        try:
                            IM = cv2.resize( cv2.imread( filename ) , (320,240)  )
                        except:
                            IM = np.zeros( (240, 320, 3) ).astype('uint8')

                        self.preload_buffer.append( IM )
                        self.preload_fnames.append( filename )



        print 'Loaded %d Items' %(len(self.preload_fnames) )




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

if __name__ == "__main__":
    PTS_BASE = 'data_Akihiko_Torii/Pitssburg/'
    pr = PittsburgRenderer( PTS_BASE )

    pr.preload_all_images( [0] )
    a, b = pr.preload_step( nP=5, nN=5)
    quit()

    a,b = pr.step(nP=5, nN=5)
    quit()
    tup = pr._query( exclude=[-1, -1, 1, -1])
    print tup
    print pr._tuple_to_filename( tup )
    sims = pr._similar_to( 10, tup )
    A = pr._get_images( sims )

    # print pr._different_than( 10, tup )
