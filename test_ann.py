""" Test `annoy` - python ANN library
    https://pypi.python.org/pypi/annoy

    Created : 24th Jan, 2017
    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
"""

from annoy import AnnoyIndex

import numpy as np


np.random.seed(1)

f = 2
t = AnnoyIndex( f, metric='euclidean' )

all_v = []

for i in range(1000):
    v = np.floor( np.random.rand(2) * 100 )
    all_v.append( v )
    t.add_item(i, v)

t.build(5) #build 5 trees


t.get_nns_by_vector( [0,0], 5, include_distances=True ) #get 5 nearest neighbour from v = [0,0]
