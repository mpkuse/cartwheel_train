""" Particle Filter / Recursive Bayesian Estimation
        Implements the Recursive Bayesian Estimation. The posterior pdf is maintained
        as particles.

        Reference:
        http://www.cim.mcgill.ca/~yiannis/particletutorial.pdf
        FAB-MAP - IJRR2008 Eq. (4)

        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 7th Mar, 2017
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import code
import argparse
from scipy.stats import rv_discrete

#
import TerminalColors
tcol = TerminalColors.bcolors()

from collections import namedtuple
Particle = namedtuple( 'Particle', 'L w') #Locations, weight

class ParticleFilter:
    def _printer( self, txt ):
        """ """
        print tcol.OKBLUE, 'ParticleFilter :', tcol.ENDC, txt

    def _error( self, txt ):
        """ """
        print tcol.FAIL, 'ParticleFilter(Error) :', tcol.ENDC, txt

    def _debug( self, txt, lvl=0 ):
        """ """
        to_print = [0,1,2]
        if lvl in to_print:
            print tcol.OKBLUE, 'ParticleFilter(Debug=%2d) :' %(lvl), tcol.ENDC, txt

    def _report_time( self, txt ):
        """ """
        print tcol.OKBLUE, 'ParticleFilter(time) :', tcol.ENDC, txt

    def gaussian( self, x, mu, sigma ):
        denom = np.sqrt( 2*np.pi*np.power(sigma,2.))
        return np.exp( -np.power(x-mu, 2.) / (2*np.power(sigma,2.))  ) #/ denom

    def dist_to_sigma( self, d ):
        # when d=0   ---> sigma=0.1 (a)
        # when d=0.3 ---> sigma=5   (b)
        a = 1.5
        b = 10
        return a + (b-a)/0.3 * d

    def __init__(self):
        M = 1000 #number of particles
        self._printer( 'Init particle filter' )
        l = np.random.uniform(low=0, high=100, size=M)
        # self.Particles = []
        self.Particle_locs = []
        self.Particle_wts  = []
        for i in range(M):
            # self.Particles.append( Particle( l[i], 1.0/float(M) ) )
            self.Particle_locs.append( l[i] )
            self.Particle_wts.append( 1.0/float(M) )
        self.Particle_locs.sort()

        # Plotting
        plt.ion()
        # self.ax = plt.gca()

    def update(self, likelihoods):
        self._printer('Update all Particles : Likelihood x Prior')

        for likes in likelihoods:
            for i in range(len(self.Particle_locs)):
                self.Particle_wts[i] *= self.gaussian( self.Particle_locs[i], likes.L, self.dist_to_sigma(likes.dist) )
            self.Particle_wts = self.Particle_wts / sum(self.Particle_wts) #normalize weights

    def resample(self):
        self._printer( 'Resample Particles. Select particles with prob proportional to their weights, according to algo in Apendix D')
        #TODO: Code here
        ran_var = rv_discrete( values=(self.Particle_locs, self.Particle_wts))
        M = len(self.Particle_locs)
        self.Particle_locs = ran_var.rvs( size=M )
        self.Particle_wts[:] = 1./M
        return


        Q = np.cumsum( self.Particle_wts )
        M = len(Q)
        t = np.random.uniform(low=0, high=1, size=M+1)
        T =np.sort( t )
        T[M] = 1.0



        i=0
        j=0
        index = []
        while i<M:
            if T[i] < Q[j]:
                index.append( j )
                i = i + 1
            else:
                j = j + 1

        tmp = np.zeros( len(self.Particle_locs) )
        tmp_wt = np.zeros( len(self.Particle_locs) )
        for i in range(len(index)):
            tmp[i] = self.Particle_locs[ index[i] ]
            tmp_wt[i] = 1.0/M #self.Particle_wts[ index[i] ]
        self.Particle_locs = tmp
        self.Particle_wts = tmp_wt
        # code.interact( local=locals() )

    def propagate(self):
        self._printer('Propagate current particles using a motion model')
        for i in range(len(self.Particle_locs)):
              self.Particle_locs[i] += 1.0 + np.random.normal()

    def plot_particles(self):
        x = []
        y = []
        zero = np.zeros( len(self.Particle_locs) )
        plt.cla()
        plt.xlim( 0,1000 )
        plt.ylim( 0,.2)
        plt.plot( self.Particle_locs, self.Particle_wts, 'b.' )
        plt.plot( self.Particle_locs, zero, 'r.')
        plt.pause(0.0001)
        # plt.show(False)
