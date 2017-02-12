""" Testing the Renderer .

    A test to try render only

    Testing soft grided class assignment
    Given x,y,z get a softdistribution of classes.
    Assuming that, grid is divided in 60x60 blocks which belong to each class.
    Probability distribution of each class is computed by fitting a 2D-gaussian
    centered at (x_,y_) with sigma=z_ * tan(40) / 3. 3*sigma should cover the circle-of-visibility
    Hidden Params : grid's blocksize (ie. 60), #of classes (ie. 120), FOV of camera (ie. 80), n_cols in grid (ie. 10)

"""

from PandaRender import TrainRenderer
import cv2
import numpy as np
import matplotlib.pyplot as plt

import SpaGrid

def gauss2D_pdf(x,y, mux, muy, sigma):
    num = (x - mux)**2 + (y-muy)**2
    return np.exp( -0.5/(sigma*sigma) * num  )


def get_soft_distribution( sg, X, Y, Z ):
    cen_array = []
    #TODO: the window (below) need to be reset as per Z.
    for x_off in range(-2,2+1):
        for y_off in range(-2,2+1):
            X_offset, Y_offset = sg.offset( X, Y, x_off, y_off )
            if X_offset is not None and Y_offset is not None:
                class_offset = sg.cord2Indx( X_offset, Y_offset )
                X_offset_cen, Y_offset_cen = sg.indx2Cord_centrum( class_offset )
                cen_array.append( (class_offset, X_offset_cen, Y_offset_cen))

                # print 'offset', class_offset, X_offset_cen, Y_offset_cen

    # Gaussian centered at X,Y with sigma= Z * np.tan(fov) / 3.0
    fov = 80
    sigma = Z * np.tan(fov/2) / 2.0
    probab = np.zeros(sg.n_classes)
    for (a_,b_,c_) in cen_array:
        print ":",a_, gauss2D_pdf( b_, c_, X, Y, sigma)
        probab[a_] =  gauss2D_pdf( b_, c_, X, Y, sigma)
    probab = probab / np.sum( probab )

    return probab


class_hot = np.zeros( (30,120) )

sg = SpaGrid.SpaGrid()
for _ in range( 100 ):
    X = np.random.uniform(0,60)
    Y = np.random.uniform(0,60)
    Z = np.random.uniform(35,120)

    class_n = sg.cord2Indx( X, Y)
    print '---\n', X,Y,Z, class_n
    probab = get_soft_distribution( sg, X, Y, Z )
    class_hot[0,:] = probab
    print probab.shape
    plt.plot( probab )
    plt.show()

    # class_n = sg.cord2Indx( X, Y )
    # X_hat, Y_hat = sg.indx2Cord( class_n)
    # X_hat_cen, Y_hat_cen = sg.indx2Cord_centrum( class_n )
    # # print X,Y, '---->', X_hat, Y_hat, '---->', X_hat_cen, Y_hat_cen
    # class_n_hat = sg.cord2Indx( X_hat, Y_hat )
    # # print class_n, class_n_hat
    # # if class_n == class_n_hat:
    # #     continue
    # # else:
    # #     break
    #
    # print '---\n', class_n, X,Y,Z
    # cen_array = []
    # for x_off in range(-2,2+1):
    #     for y_off in range(-2,2+1):
    #         X_offset, Y_offset = sg.offset( X, Y, x_off, y_off )
    #         if X_offset is not None and Y_offset is not None:
    #             class_offset = sg.cord2Indx( X_offset, Y_offset )
    #             X_offset_cen, Y_offset_cen = sg.indx2Cord_centrum( class_offset )
    #             cen_array.append( (class_offset, X_offset_cen, Y_offset_cen))
    #
    #             print 'offset', class_offset, X_offset_cen, Y_offset_cen
    #
    # # Gaussian centered at X,Y with sigma= Z * np.tan(fov) / 3.0
    # fov = 80
    # sigma = Z * np.tan(fov/2) / 2.0
    # probab = np.zeros(sg.n_classes)
    # for (a_,b_,c_) in cen_array:
    #     print ":",a_, gauss2D_pdf( b_, c_, X, Y, sigma)
    #     probab[a_] =  gauss2D_pdf( b_, c_, X, Y, sigma)
    # probab = probab / np.sum( probab )



quit()




app = TrainRenderer()
#app.run()

t = 0
while True:
    # im_batch, label_batch = app.step( 20 )
    # for i in range(20):
    #     fname = 'dump/'+str(t)+'.jpg'
    #     # print 'Write file : ',fname
    #     # cv2.imwrite( fname, im_batch[i,:,:,:] )
    #
    #     x_ = label_batch[i,0]
    #     y_ = label_batch[i,1]
    #     z_ = label_batch[i,2]
    #     CLASS = np.floor(x_/60) + np.floor(y_/60)*10  + 65
    #     print "CLASS",CLASS
    #
    #     t = t + 1


    app.taskMgr.step()
