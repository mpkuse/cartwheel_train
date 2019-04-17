""" Demo usage script for my various renderers.

    These renderer classes are supposed to be used for feeding data into
    tensorflow for training. This gives a nice way to separate data-handling
    from the core learning part. This script demos the usage of the rendering
    classes. This is mainly done because in my NetVLAD training I need to
    draw complicated sample (query, positive set, negative set). This is very
    different from the standard image-net style training where you just need
    to feed images independently.

    Currently has the following renderers.
    a) PandaRender - Images from 3d model. (needs to have python package Panda3d)
    b) PittsburgRenderer - Draws samples from Pittsburg dataset
    c) WalksRenderer - Keezi_walks is a youtube channel of walking around various cities with a camera. I have collected a few of those videos.
    d) TimeMachineRender - From tokyoTM dataset


        Author  : Manohar Kuse <mpkuse@connect.ust.hk>
        Created : 11th July, 2017
        Updated : 7th Feb, 2018
"""

import numpy as np
import cv2
import time
import code

from TimeMachineRender import TimeMachineRender
#from PandaRender import NetVLADRenderer
from WalksRenderer import WalksRenderer
from WalksRenderer import WalksRendererOnline
from PittsburgRenderer import PittsburgRenderer

def demo_pittsburg():
    """ Uses the Pitssburg data set. To obtain this dataset make a
    request here: http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/

    Some of the older Google StreetView data does not have compasss info
    so beware.

    """
    PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
    pr = PittsburgRenderer( PTS_BASE )
    for i in range(20):
        a,b = pr.step(nP=10, nN=10, ENABLE_IMSHOW=True, return_gray=False, resize=(160, 120))
        # print a.shape
        # print b.shape
        # code.interact(local=locals() )
        cv2.waitKey(0)
    quit()

def demo_walks():
    """ Although this works, but using this with training is currently
        too slow. The problem is it takes way lot time to seek a video
        file. May be in the future will work to improve this renderer.

        Keezi_walks youtube channel: https://www.youtube.com/user/keeezi
        Contact me (mpkuse@connect.ust.hk), I can provide a few videos.

        This class needs work. Something on the lines of
        preloading image frames of the videos.
    """
    WR_BASE = './keezi_walks/'
    WR_BASE = '/media/mpkuse/Bulk_Data/keezi_walks/'

    wr = WalksRenderer( WR_BASE )
    for i in range(20):
        a,b = wr.step(nP=10, nN=10)
        print a.shape
        print b.shape
        cv2.waitKey(0)
    quit()


def demo_tokyotm():
    """ Needs TokyoTM dataset.
        Request here: http://www.ok.ctrl.titech.ac.jp/~torii/project/247/
    """
    TTM_BASE = '/Bulk_Data/data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
    tm = TimeMachineRender( TTM_BASE )
    pyDB = tm.pyDB
    # tm.debug_display_image_samples()
    for i in range(100):
        a,b = tm.step(5,5, return_gray=True, ENABLE_IMSHOW=True)
        print 'a.shape=', a.shape, '\tb.shape=', b.shape
        cv2.waitKey(0)

def demo_panda():
    """ Needs a 3d Model and some knowhow of Panda3d graphics rendering engine.
    I (mpkuse@connect.ust.hk) can provide 3d model for HKUST constructed using
    the Altizure system (https://www.altizure.com/) from a drone video.
    """

    app = NetVLADRenderer()
    for i in range(20):
        a,b = app.step(16)

demo_pittsburg()
# demo_walks()
# demo_tokyotm()
# demo_panda()


# WALKS_PATH = '/media/mpkuse/Bulk_Data/keezi_walks/'
# tm = WalksRendererOnline( WALKS_PATH )
# tm.proc()
