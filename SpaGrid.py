""" Defines a class SpaGrid.
        This class mainly provides function to get class ID given the x,y cord.

"""

import numpy as np


class SpaGrid:

    ## 0,  1 , 2 , 3 , ... 10
    ## 11, 12, 13, ...     20
    ## 21, 22, 23, ...     30
    ## .


    ## Given x,y (origin in middle) return class indx
    def cord2Indx( self, x_, y_ ):
        #convert to top-left cord system
        x_tl = x_ - self.x_min
        y_tl = -(y_ - self.y_max)

        x_nc = np.floor( x_tl / self.x_box_size )
        y_nc = np.floor( y_tl / self.y_box_size )

        class_n = x_nc + y_nc * self.n_boxes_x
        return int(class_n)

    ## Given a class index, return the co-ordinate of the top-left corner of the cell.
    ## Return co-ordinate are in co-ordinate system of the grid, ie. NOT in top-left cord system
    def indx2Cord( self, class_n ):
        x_nc = class_n % self.n_boxes_x
        y_nc = np.floor(class_n / self.n_boxes_x)

        x_nc = x_nc * self.x_box_size
        y_nc = y_nc * self.y_box_size

        x_ = x_nc  + self.x_min
        y_ = -y_nc + self.y_max

        return x_,y_

    ## same as `indx2Cord` but returns co-ordinates of the center of the cell
    def indx2Cord_centrum( self, class_n ):
        x_, y_ = self.indx2Cord( class_n )
        x_cen = x_ + self.x_box_size/2.0
        y_cen = y_ - self.x_box_size/2.0
        return x_cen, y_cen


    def  offset( self, X_, Y_, xoffset, yoffset ):
        X_o = X_ + xoffset*self.x_box_size
        Y_o = Y_ + yoffset*self.y_box_size
        if X_o > self.x_max or X_o < self.x_min or  Y_o > self.y_max or Y_o < self.y_min :
            return None, None
        else:
            return X_o, Y_o

    def __init__(self):
        self.n_boxes_x = 10 # number of boxes in x-dir
        self.n_boxes_y = 12

        self.x_min = -300
        self.x_max = 300
        self.y_min = -360
        self.y_max = 360
        self.z_min = 35
        self.z_max = 120

        self.x_box_size = (self.x_max - self.x_min)/self.n_boxes_x # length of x-side of the box
        self.y_box_size = (self.y_max - self.y_min)/self.n_boxes_y # length of y-side of the box

        self.n_classes = self.n_boxes_x * self.n_boxes_y #number of classes
