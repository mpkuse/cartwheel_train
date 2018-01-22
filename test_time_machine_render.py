""" testing for TimeMachineRender.TimeMachineRender """

import numpy as np
import cv2
import time
import code

from TimeMachineRender import TimeMachineRender
from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from PittsburgRenderer import PittsburgRenderer

PTS_BASE = '/media/mpkuse/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
pr = PittsburgRenderer( PTS_BASE )
for i in range(20):
    a,b = pr.step(nP=10, nN=10)
    print a.shape
    print b.shape
    cv2.waitKey(0)
quit()

WR_BASE = './keezi_walks/'
wr = WalksRenderer( WR_BASE )
a,b = wr.step(nP=10, nN=10)
print a.shape
print b.shape
cv2.waitKey(0)
quit()


TTM_BASE = 'data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
tm = TimeMachineRender( TTM_BASE )
pyDB = tm.pyDB
# tm.debug_display_image_samples()
for i in range(100):
    a,b = tm.step(5,5, return_gray=True)
    cv2.waitKey(0)
    code.interact( local=locals() )
    # a,b = tm.step_random(7)

# app = NetVLADRenderer()
# a,b = app.step(16)
# code.interact( local=locals() )
