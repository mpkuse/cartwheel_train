from collections import namedtuple
from collections import deque
import random
import numpy as np
import heapq
import code


#
import TerminalColors
tcolor = TerminalColors.bcolors()

class NDPoint(object):
    """
    A point in n-dimensional space
    """

    def __init__(self, x, idx=None):
        self.x = np.array(x)
        self.idx = idx
    def __repr__(self):
        return "NDPoint(idx=%s, x=%s)" % (self.idx, self.x)

class VPTree(object):
    """
    An efficient data structure to perform nearest-neighbor
    search.
    """

    def __init__(self, points, dist_fn=None):
        self.left = None
        self.right = None
        self.mu = None
        self.dist_fn = dist_fn if dist_fn is not None else l2

        # choose a better vantage point selection process
        self.vp = points.pop(random.randrange(len(points)))

        if len(points) < 1:
            return

        # choose division boundary at median of distances
        distances = [self.dist_fn(self.vp, p) for p in points]
        self.mu = np.median(distances)

        left_points = []  # all points inside mu radius
        right_points = []  # all points outside mu radius
        for i, p in enumerate(points):
            d = distances[i]
            if d >= self.mu:
                right_points.append(p)
            else:
                left_points.append(p)

        if len(left_points) > 0:
            self.left = VPTree(points=left_points, dist_fn=self.dist_fn)

        if len(right_points) > 0:
            self.right = VPTree(points=right_points, dist_fn=self.dist_fn)

    def is_leaf(self):
        return (self.left is None) and (self.right is None)





class PriorityQueue(object):
    def __init__(self, size=None):
        self.queue = []
        self.size = size

    def push(self, priority, item):
        self.queue.append((priority, item))
        self.queue.sort()
        if self.size is not None and len(self.queue) > self.size:
            self.queue.pop()


class InnerProductTree(object):
    """ These trees are difined on an innerproduct spaces,
        thus every elements should be unit vectors.
        Distance = sqrt(1 - <a,b>). DIstance \in [0,sqrt(2)]


        These trees have a fixed threshold on mu and can
        on be constructed online. Another possibility is to
        add say 100 initial points in bulk and then add incrementally"""

    def __init__(self, p, th_mu=0.2):
        self.left = None
        self.right = None
        self.mu = th_mu
        self.vp = p
        self.dist_fn = dot

    def add_item( self, f ):
        is_added = False
        cur = self
        dis = dot( f, cur.vp )

        while is_added == False:
            if  dis > cur.mu:
                #Add to right sub-tree
                if cur.right is not None:
                    cur = cur.right
                else:
                    cur.right = InnerProductTree(f, self.mu)
                    is_added = True
            elif dis < cur.mu:
                #Add to left sub-tree
                if cur.left is not None:
                    cur = cur.left
                else:
                    cur.left = InnerProductTree(f, self.mu)
                    is_added = True
            else:
                #equal to : not adding
                print tcolor.WARNING, 'Same as cur, not adding element'
                is_added = True


    def is_leaf(self):
        return (self.left is None) and (self.right is None)







### Distance functions
def l2(p1, p2):
    return np.sqrt(np.sum(np.power(p2.x - p1.x, 2)))


def dot(p1,p2):
    """ Dot product distance metric. inputs must be unit vectors"""
    n = np.maximum(0.0, (1.0 - np.dot( p1.x, p2.x)) )
    return np.sqrt(n)

def visualize( tree, lvl=0, VERBOSE=False, max_lvl=10000000 ):
    space = '    '
    stack = [tree]
    stack_lvl = [0]
    stack_chh = ['v']

    # Tree Statistics
    all_lvl = []
    n_2_children = 0
    n_1_children = 0
    n_left_only = 0
    n_right_only = 0
    n_leafs = 0
    n_total = 0


    while len(stack) > 0 :
        cur = stack.pop()
        n_total += 1
        cur_lvl = stack_lvl.pop()
        cur_chh = stack_chh.pop()
        all_lvl.append( cur_lvl )

        # code.interact( local=locals())
        if max_lvl > cur_lvl:
            if VERBOSE:
                print space*cur_lvl, cur_chh, np.round( list(cur.vp.x) , 2), cur.vp.idx, tcolor.OKBLUE, 'mu=', np.round(cur.mu,2) if cur.mu is not None else 'None', tcolor.ENDC # np.round( list(cur.vp.x) , 2), ':idx=%3d:mu=%4.2f' %(cur.vp.idx, cur.mu)
            else:
                if cur_lvl > 0:
                    print 'lvl=%3d' %(cur_lvl), space*(cur_lvl-1)+'  %c-' %(cur_chh), '%04d' %(cur.vp.idx), tcolor.OKBLUE, 'mu=', np.round(cur.mu,2) if cur.mu is not None else 'N/A', tcolor.ENDC
                else:
                    print 'lvl=%3d' %(cur_lvl), '%04d' %(cur.vp.idx), tcolor.OKBLUE, 'mu=', np.round(cur.mu,2) if cur.mu is not None else 'N/A', tcolor.ENDC

        if cur.left is not None:
            stack.append( cur.left )
            stack_lvl.append( cur_lvl+1 )
            stack_chh.append( 'l')

        if cur.right is not None:
            stack.append( cur.right )
            stack_lvl.append( cur_lvl+1 )
            stack_chh.append( 'r')

        # Collect statistics
        if cur.left is None and cur.right is None: #None, None --> leaf
            n_leafs += 1
        if cur.left is None and cur.right is not None: #None, None --> leaf
            n_right_only += 1
            n_1_children += 1
        if cur.left is not None and cur.right is None: #None, None --> leaf
            n_left_only += 1
            n_1_children += 1
        if cur.left is not None and cur.right is not None: #None, None --> leaf
            n_2_children += 1



    print '--- STATISTICS ---'
    print 'INFO: Max Depth : ', max(all_lvl)
    print 'n_2_children', n_2_children
    print 'n_1_children',n_1_children
    print 'n_left_only', n_left_only
    print 'n_right_only',n_right_only
    print 'n_leafs',n_leafs
    print 'n_total', n_total
    print '--- --  END -- ---'

### Operations
def get_nearest_neighbors(tree, q, k=1):
    """
    find k nearest neighbor(s) of q

    :param tree:  vp-tree
    :param q: a query point
    :param k: number of nearest neighbors

    """

    # buffer for nearest neightbors
    neighbors = PriorityQueue(k)

    # list of nodes ot visit
    visit_stack = deque([tree])

    # distance of n-nearest neighbors so far
    tau = np.inf

    while len(visit_stack) > 0:
        node = visit_stack.popleft()
        if node is None:
            continue

        d = tree.dist_fn(q, node.vp)
        if d < tau:
            neighbors.push(d, node.vp)
            tau, _ = neighbors.queue[-1]

        if node.is_leaf():
            continue

        if d < node.mu:
            if d < node.mu + tau:
                visit_stack.append(node.left)
            if d >= node.mu - tau:
                visit_stack.append(node.right)
        else:
            if d >= node.mu - tau:
                visit_stack.append(node.right)
            if d < node.mu + tau:
                visit_stack.append(node.left)
    return neighbors.queue


def get_all_in_range(tree, q, tau):
    """
    find all points within a given radius of point q

    :param tree: vp-tree
    :param q: a query point
    :param tau: the maximum distance from point q
    """

    # buffer for nearest neightbors
    neighbors = []

    # list of nodes ot visit
    visit_stack = deque([tree])

    while len(visit_stack) > 0:
        node = visit_stack.popleft()
        if node is None:
            continue

        d = tree.dist_fn(q, node.vp)
        if d < tau:
            neighbors.append((d, node.vp))

        if node.is_leaf():
            continue

        if d < node.mu:
            if d < node.mu + tau:
                visit_stack.append(node.left)
            if d >= node.mu - tau:
                visit_stack.append(node.right)
        else:
            if d >= node.mu - tau:
                visit_stack.append(node.right)
            if d < node.mu + tau:
                visit_stack.append(node.left)
    return neighbors



if __name__ == '__main__':
    # np.random.seed(0)
    # X = np.random.uniform(10, 100, size=60)
    # Y = np.random.uniform(10, 100, size=60)
    X = np.random.randn( 60)
    Y = np.random.randn( 60)
    points = [NDPoint(x/np.linalg.norm(x),i) for i, x in  enumerate(zip(X,Y))]

    # tree = VPTree(points)
    # q = NDPoint([80,55])
    # neighbors = get_nearest_neighbors(tree, q, 5)
    #
    # print "query:"
    # print "\t", q
    # print "nearest neighbors: "
    # for d, n in neighbors:
    #     print "\t", n
    #
    # visualize( tree, VERBOSE=True )

    tree = InnerProductTree(points[0], 0.5)
    for i in range(1,len(points)):
        print 'Add Item :', points[i]
        tree.add_item(points[i])
    visualize( tree, VERBOSE=True )
