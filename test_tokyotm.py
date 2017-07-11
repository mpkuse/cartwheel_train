import scipy.io
import numpy as np
#import matplotlib.pyplot as plt
import pyqtgraph as pg
import time
import cv2

TTM_BASE = 'data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM

mat = scipy.io.loadmat( TTM_BASE+'/tokyoTM_train.mat' )
dbStruct = mat['dbStruct']
utmQ = dbStruct['utmQ'].item()
utmDb = dbStruct['utmDb'].item() #2x...

dbImageFns = dbStruct['dbImageFns'].item()
qImageFns = dbStruct['qImageFns'].item()
#can do dbImageFns[0][0][0], dbImageFns[1][0][0], dbImageFns[2][0][0], dbImageFns[3][0][0] , ...

dbTimeStamp = dbStruct['dbTimeStamp'].item()[0,:]
qTimeStamp = dbStruct['qTimeStamp'].item()[0,:]

#pw = pg.plot()
pyDB = {}
for i in range( utmQ.shape[1] ):
    print 'Db', utmDb[0,i], utmDb[1,i]#, dbTimeStamp[i]
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




    #pw.plot( utmDb[0,i:i+1], utmDb[1,i:i+1], pen=None, symbol='o' )
    #cv2.imshow( 'win', cv2.imread( 'images/'+dbImageFns[i][0][0] ) )

    #print 'Q ', utmQ[0,i], utmQ[1,i], qTimeStamp[i]#, qImageFns[i][0][0]
    #pw.plot( utmQ[0,i:i+1], utmQ[1,i:i+1], pen=None, symbol='s' )
    #cv2.imshow( 'win', cv2.imread( 'images/'+qImageFns[i][0][0] ) )

    #pg.QtGui.QApplication.processEvents()

    #cv2.waitKey(0)


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

			cv2.imshow( str(y)+'_'+str(circ), cv2.resize(cv2.imread( TTM_BASE+'/images/'+pyDB[l][y][circ] ), (0,0), fx=0.25, fy=0.25 ) )
			cv2.moveWindow( str(y)+'_'+str(circ), 180*yi, 10+180*circ_i )
			print 200*yi, 10+200*circ_i

			win_list.append( str(y)+'_'+str(circ) )

	cv2.waitKey(0)
	for w in win_list:
		cv2.destroyWindow(w)
