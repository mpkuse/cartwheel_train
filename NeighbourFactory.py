""" Defines the class Neighbour Factory
        This class loads the file-dump containing matrix M_{Nx8192} with N
        sample descriptors. thumbs stored for debugging.
"""
import numpy as np
import cv2
from annoy import AnnoyIndex
import time
import code


#
import TerminalColors
tcolor = TerminalColors.bcolors()



class NeighbourFactory:
    def __init__(self, PARAM_DESCRIPTOR_DB ):
        print 'Load : ', PARAM_DESCRIPTOR_DB
        npz_data = np.load( PARAM_DESCRIPTOR_DB )
        self.M = npz_data['M'] # N x 8192. N is number of instances
        self.thumbs = npz_data['thumbs'] # Nx48x64x3. corresponding thumbnails
        print tcolor.OKGREEN, 'M.shape=', self.M.shape, tcolor.ENDC
        startTime = time.time()
        print 'Building KD-Tree'
        self._build_kdtree()
        print tcolor.OKBLUE, 'Built KDtree in %4.2fs with n_items=%d, each of dim=%d' %(time.time()-startTime, self.kdtree.get_n_items(), len(self.kdtree.get_item_vector(0)) ), tcolor.ENDC

        self._full_set = set(range(0,self.M.shape[0]))



    def _build_kdtree(self):
        self.kdtree = AnnoyIndex(self.M.shape[1], metric='angular') #sqrt[  2(1-cos(u,v))   ]
        for u in range(self.M.shape[0]):
            self.kdtree.add_item(u, self.M[u,:] )
        self.kdtree.build(10)

    def _debug( self, msg ):
        """ """
        # print tcolor.OKBLUE, '[DEBUG]', msg, tcolor.ENDC


    ## Get n neighbours of i^{th} element in KDtree. Note that this may return less than n
    ## neighbours based on thresh
    def get_neighbours( self, i, n, thresh ):
        self._debug( 'find %d neighbours of %d^{th} item' %(n, i) )
        nn, nn_dist = self.kdtree.get_nns_by_item( i, n, include_distances=True )
        nn = np.array(nn)
        nn_dist = np.array(nn_dist)

        self._debug( 'retain neighbours whose angular-distance is less than %4.3f' %(thresh) )
        nn_indx = nn_dist < thresh

        self._debug( '%d feasible neighbours' %( (nn_indx==True).sum() ) )

        feasible_neighbours = nn[ np.where( nn_dist < thresh ) ]
        return feasible_neighbours


    def get_non_neighbours( self, i, n ):
        # feasible_neighbours = get_neighbours( kdtree, i, n, thresh )
        feasible_neighbours = self.kdtree.get_nns_by_item( i, 2*n )
        f = set(feasible_neighbours)
        diff_set = self._full_set - f
        diff_set_ary = np.array([ int(x) for x in diff_set])

        rand_i = np.random.randint(0,len(diff_set), n ) #n random integers in (0,N)
        to_return = diff_set_ary[rand_i]

        return to_return


    def dot( self, i, j):
        return np.dot( self.M[i,:], self.M[j,:] )
