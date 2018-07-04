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
import code
import random
import os
import pickle
import datetime
#
import TerminalColors
tcolor = TerminalColors.bcolors()

import networkx as nx
import matplotlib.pyplot as plt


class WeightedRandomizer:
    """ Class to give weighted random numbers

    adopted from : https://stackoverflow.com/questions/14992521/python-weighted-random

    Sample usage:
    w = {'A': 1.0, 'B': 1.0, 'C': 18.0}
    wr = WeightedRandomizer (w)
    print wr.random ()

    """

    def __init__ (self, weights):
        self.__max = .0
        self.__weights = []
        for value, weight in weights.items ():
            self.__max += weight
            self.__weights.append ( (self.__max, value) )

    def random (self):
        r = random.random () * self.__max
        for ceil, value in self.__weights:
            if ceil > r: return value


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


class WalksRendererPreload:
    def __init__( self, db_path, FROM_PICKLE = True, CACHE_PICKLES=False, list_of_video_files=None ):
        """ This will load all the videos along with loop closure data as undirected graph.
            list_of_video_files : if none, means all

            if FROM_PICKLE is true, than list_of_video_files will be ignored
        """
        self.db_path = db_path
        print tcolor.OKGREEN, 'WalksRenderer.db_path : ', db_path, tcolor.ENDC

        # N is number of videos. K is the number of key frames for each video,
        # L is the number of loop-events in that particular video
        self.frames = [] #NxKx240x320x3
        self.frame_ids = [] #NxK
        self.graphs = []

        # Load from pickles
        self.BBB = db_path #'/media/mpkuse/Bulk_Data/scratch_pad/'


        if FROM_PICKLE:
            print 'READ ', self.BBB+'self.frames.pickle'
            print 'READ ', self.BBB+'self.frame_ids.pickle'
            print 'READ ', self.BBB+'self.graphs.pickle'
            self.frames = pickle.load(  open( self.BBB+'self.frames.pickle', 'rb' ) )
            self.frame_ids = pickle.load(  open( self.BBB+'self.frame_ids.pickle', 'rb' ) )
            self.graphs = pickle.load(  open( self.BBB+'self.graphs.pickle', 'rb' ) )
        else:
            print tcolor.OKBLUE, 'Video Files : ', tcolor.ENDC
            if list_of_video_files is None:
                list_of_video_files = glob.glob( db_path+"/*.mp4" ) + glob.glob( db_path+"/*.webm" ) +glob.glob( db_path+"/*.mkv" )
            # list_of_video_files =  glob.glob( db_path+"/Waterfall_drone.mp4" ) + glob.glob( db_path+"/Vegas_night_drone.mp4" ) + glob.glob( db_path+"/Windmills_drone.mp4" )
            self.load_video_files( list_of_video_files, save_intermediate_to_pickle=CACHE_PICKLES )


        # Graph choicer (weighted random number generation)
        _weighting = {}
        for ggg in range( len(self.graphs) ):
            _weighting[ str(ggg) ] = len(self.graphs[ggg])
        self.wr_2 = WeightedRandomizer( _weighting )

        # # Remove this code and the pickle files after testing is done
        # # Save as pickles
        if CACHE_PICKLES:
            fp = open( self.BBB+'self.frames.pickle', 'wb' )
            pickle.dump( self.frames, fp )
            fp.close()
            fp = open( self.BBB+'self.frame_ids.pickle', 'wb' )
            pickle.dump( self.frame_ids, fp )
            fp.close()
            fp = open( self.BBB+'self.graphs.pickle', 'wb' )
            pickle.dump( self.graphs, fp )
            fp.close()






    def _similar(self, graph_idx):
        # Get similars

        # G.degree() #degrees of all nodes
        # G.adj() #adjacent nodes of every node

        G = self.graphs[graph_idx] # select a graph
        I = self.frames[graph_idx]
        Ii = self.frame_ids[graph_idx]

        wr = WeightedRandomizer( G.degree() )
        n = wr.random()
        #n = random.choice( G.nodes() ) # select a node

        # print 'degree of node=%d is %d' %(n, G.degree()[n])
        # print 'adj of n=%d are : %s' %( n, str(G.adj[n].keys()) )
        L = []
        for hn in G.adj[n].keys(): # loop over neighbours of n
            L = L + [hn] + G.adj[hn].keys()
            for hn1 in G.adj[hn].keys(): #loop over neighbours of neighbours of n
                L = L + [hn1] + G.adj[hn1].keys()

        L = sorted(set(L))
        # L is not sorted uniq set of adj(adj(adj(n)))

        return L


    def step( self, nP, nN, apply_distortions=True, return_gray=False, ENABLE_IMSHOW=False):

        # Choose a graph randomly to make similar choices


        # unweighted graph choice
        #graph_idx = random.choice( range(len(self.graphs) ) )

        # weighted graph choice, proportional to number of frames in a graph
        graph_idx = int(self.wr_2.random())
        # code.interact( local=locals() )


        L = self._similar( graph_idx )

        # Choose any nP+1 from set L of graph[graph_idx]
        I = self.frames[graph_idx]
        Ii = self.frame_ids[graph_idx]
        # L_sampled = random.sample( L, nP+1 )

        intersection_set = set(Ii).intersection( set(L) )
        if len(intersection_set) >= (nP + 1):
            L_sampled = random.sample( intersection_set, nP+1 )
        else:
            # Trying to sample more than there are elements
            print 'Trying to sample more than there are elements'
            L_sampled = random.sample( intersection_set, len(intersection_set) ) +\
                        random.sample( intersection_set, nP+1-len(intersection_set ) )
            code.interact( local=locals() )

        SSSS = [] # similar images list
        for l in L_sampled:
            # print graph_idx, l
            try:
                _IM =  I[ Ii.index(l) ]
            except:
                code.interact( banner="choose +ve list exception", local=locals() )
            SSSS.append( _IM )
            # cv2.imshow( 'win', I[ Ii.index(l) ] )
            # cv2.waitKey(0)


        # code.interact( local=locals() )
        # Now choose negative set
        # Choose any nN (disimilar) not in set L
        DDDD = []
        for j in range( nN ):
            # Choose graph
            graph_idx2 = random.choice( range(len(self.graphs) ) )
            _I = self.frames[graph_idx2]
            _Ii = self.frame_ids[graph_idx2]

            # Choose a node from this graph. Avoid last 20 items
            n = random.choice( self.graphs[graph_idx2].nodes() )

            while True: # index not in list choose again
                # print graph_idx2, n
                try:
                    _IM =  _I[ _Ii.index(n) ]
                    break
                except:
                    # print '.',
                    n = random.choice( self.graphs[graph_idx2].nodes() )
                    # code.interact( banner="choose -ve list exception", local=locals() )
            DDDD.append( _IM )
            # cv2.imshow( 'win', _I[ _Ii.index(n) ] )
            # cv2.waitKey(0)

        ######## Ready with SSSS, DDDD. Next apply distortion

        OUT_S = []
        for IM in SSSS:
            if apply_distortions:
                # Intensity transform
                gamma = np.random.rand() + 0.5
                adjusted = adjust_gamma( IM, gamma=gamma )
                IM = adjusted

            if apply_distortions == True and np.random.rand() > 0.75:
                # Planar rotation, cropped. adopted from `test_rot-test.py`
                image_height, image_width = IM.shape[0:2]
                image_orig = np.copy(IM)
                irot = np.random.uniform(-90,90 )#np.random.randn() * 25.
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

            if return_gray == True:
                IM_gray = cv2.cvtColor( IM, cv2.COLOR_BGR2GRAY )
                IM = np.expand_dims( IM_gray, axis=2 )

            OUT_S.append( IM )


        OUT_D = []
        for IM in DDDD:
            if apply_distortions:
                # Intensity transform
                gamma = np.random.rand() + 0.5
                adjusted = adjust_gamma( IM, gamma=gamma )
                IM = adjusted

            if apply_distortions and np.random.rand() > 0.75:
                # Planar rotation, cropped. adopted from `test_rot-test.py`
                image_height, image_width = IM.shape[0:2]
                image_orig = np.copy(IM)
                irot = np.random.uniform(-90,90 )#np.random.randn() * 25.
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

            if return_gray == True:
                IM_gray = cv2.cvtColor( IM, cv2.COLOR_BGR2GRAY )
                IM = np.expand_dims( IM_gray, axis=2 )

            OUT_D.append( IM )





        if ENABLE_IMSHOW:
            cv2.imshow( 'similar', np.concatenate(np.array(OUT_S), axis=1) )
            cv2.imshow( 'dissimilar', np.concatenate(np.array(OUT_D), axis=1) )
            cv2.waitKey(1)

        return np.array( OUT_S + OUT_D ).astype('float32'), np.zeros( (1+nP+nN,4) )




    def load_video_files( self, list_of_files, clear_data=False, save_intermediate_to_pickle=True ):
        """ Given a list of files popilates self.frames, self.frame_ids, self.graphs.
            This function is kept separate. This logic was for larger dataset it
            will be impossible to load everything. So say load 20 videos at a
            time. Run a few 1000 iterations of neural net training. Then load
            say more videos etc.
        """

        if clear_data:
            self.frames = []
            self.frame_ids = []
            self.graphs = []

        print 'Video Files: '
        for _i, file_name in enumerate( list_of_files ):
            print tcolor.OKBLUE, _i, '.', file_name , tcolor.ENDC

        for _i, file_name in enumerate( list_of_files ):

            # Make a pose-graph (pseudo) for this video.
            print '[%s]' %(str(datetime.datetime.now())), _i, tcolor.OKBLUE, '+', file_name, tcolor.ENDC

            fr, fr_id, G = self._make_undirected_graph( file_name )
            # nx.draw_circular( G, with_labels=True )
            # plt.show()


            self.frames.append( fr )
            self.frame_ids.append( fr_id )
            self.graphs.append( G )

            if save_intermediate_to_pickle:
                # Remove this code and the pickle files after testing is done
                # # Save as pickles
                print 'Save: ', self.BBB+'self.frames.pickle'
                print 'Save: ', self.BBB+'self.frame_ids.pickle'
                print 'Save: ', self.BBB+'self.graphs.pickle'
                fp = open( self.BBB+'self.frames.pickle', 'wb' )
                pickle.dump( self.frames, fp )
                fp.close()
                fp = open( self.BBB+'self.frame_ids.pickle', 'wb' )
                pickle.dump( self.frame_ids, fp )
                fp.close()
                fp = open( self.BBB+'self.graphs.pickle', 'wb' )
                pickle.dump( self.graphs, fp )
                fp.close()


    def _make_undirected_graph( self, file_name ):
        """
            Read the video file `file_name` and load all frames as images.
            Also read loop-closure txt file `file_name`+.txt.
            Return a) list of frames, b) like of position index in video
            c) undirected graph (networkx) G.
        """

        # Load 1/10 frames
        fr, fr_id = self._preload_video( file_name )

        # Make Graph - Odometry edges
        G = nx.Graph()
        G.add_nodes_from( fr_id )

        # Odometry edges
        fr_odom = self._make_odom_edges( fr_id, 2 )  #[ (a,b), (a,b), ... ]
        G.add_edges_from( fr_odom  )


        # Load loop-closure file. It has 3 cols. 1st col is curr frame index, 2nd col prev is frame index, 3rd col is nInliers
        print 'Open Loop Closure : ', file_name+'.txt'
        if os.path.isfile( file_name+'.txt' ):
            l_info = np.loadtxt( file_name+'.txt' , dtype='int32', delimiter=',')
            print 'Found. Contains %d loop messages' %(l_info.shape[0])
            loop_edges = self._dbowinfo_to_edgelist( l_info )
            G.add_edges_from( loop_edges  )
        else:
            print 'Not Found'

        return fr, fr_id, G


    def _preload_video( self, file_name, skip=10 ):
        # loop thru the video skip 10 frames
        FR = []
        FR_ID = []
        print 'Open Video: ', file_name
        cap = cv2.VideoCapture( file_name )
        # code.interact( local=locals() )
        i = -1
        nFrames = cap.get( cv2.CAP_PROP_FRAME_COUNT )
        print 'Preload Video : Total Frames : %d' %( nFrames )
        print tcolor.OKGREEN, 'Loop Running', tcolor.ENDC
        while cap.isOpened() and i < (nFrames-20):
            i = i+1
            try:
                ret, frame = cap.read()
            except:
                code.interact( local=locals(), banner="_preload_video" )
                continue
            if i%skip != 0:
                continue
            else:
                if i%(100*skip) == 0:
                    print 'frame#%06d of %06d for file:%s' %(i, nFrames, file_name )
            	frame_sm = cv2.resize( cv2.blur(frame, (5,5)), (320,240) )
                FR.append( frame_sm )
                FR_ID.append( i )
            	# print frame_sm.shape
            	# cv2.imshow( 'frame', frame_sm )
            	# if( cv2.waitKey( 10 ) & 0xFF ) == ord('q'):
            		# break

        cap.release()
        return FR, FR_ID

    def _dbowinfo_to_edgelist( self, l_info ):
        """ l_info is Nx3. 1st col is node-1, 2nd col is node-2.
        node-1, node-2 represent a loop closure event as determined by
        another external program.

        returns [(e1,e2), (e3,e4), ... ]
        """

        #TODO: Take also the nodes of the graph as input for this function.
        #      Only return edges which form part of this graph. ie. Dont
        #      append an edge if it is not existant in the graph G.

        E = []
        for l in l_info:
            E.append( (l[0], l[1]) )
        return E

    def _make_odom_edges( self, fr_id, k ):
        """
            fr_id : Kx1
            k     : connections to # of prev nodes
              [ (a,b), (a,b), ... ]
        """

        odom_edges = [ (fr_id[1], fr_id[0]) ]
        for i in range( 2, len(fr_id) ):
            # print '---'
            # print fr_id[i], '<-odom->', fr_id[i-1]
            # print fr_id[i], '<-odom->', fr_id[i-2]
            odom_edges.append(   (fr_id[i], fr_id[i-1])    )
            odom_edges.append(   (fr_id[i], fr_id[i-2])    )
        return odom_edges







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

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)
