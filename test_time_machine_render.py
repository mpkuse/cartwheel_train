""" testing for TimeMachineRender.TimeMachineRender """

import numpy as np
import cv2
import time
import code

from TimeMachineRender import TimeMachineRender
from PandaRender import NetVLADRenderer

TTM_BASE = 'data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
tm = TimeMachineRender( TTM_BASE )
pyDB = tm.pyDB
# tm.debug_display_image_samples()
for i in range(100):
    a,b = tm.step(5,5)
    cv2.waitKey(0)
    # a,b = tm.step_random(7)

# app = NetVLADRenderer()
# a,b = app.step(16)
# code.interact( local=locals() )
