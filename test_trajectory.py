""" Test the renderer in trajectory mode """


from PandaRender import TestRenderer
import cv2
import numpy as np
import matplotlib.pyplot as plt



app = TestRenderer()

l = 0
while True:
    im, y = app.step()

    if im is not None:
        # cv2.imwrite( 'dump/'+str(l)+'.jpg', im )

        l = l + 1
