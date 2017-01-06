#Parameters
#----------
#image : ndarrayz
#    Input image data. Will be converted to float.
#mode : str
#    One of the following strings, selecting the type of noise to add:
#
#    'gauss'     Gaussian-distributed additive noise.
#    'poisson'   Poisson-distributed noise generated from the data.
#    's&p'       Replaces random pixels with 0 or 1.
#    'speckle'   Multiplicative noise using out = image + n*image,where
#                n,is uniform noise with specified mean & variance.

import numpy as np
import os
import cv2
import copy

from skimage import data, exposure, img_as_float

def intensity_transform(im_gray):
    rand_flt = (np.random.random() * 1.5) + 0.5
    gamma_corrected = exposure.adjust_gamma(im_gray, rand_flt)
    return gamma_corrected



def noisy(noise_typ,image):

    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 7
       #sigma = var**0.5
        gauss = np.random.normal(mean,var,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = copy.deepcopy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy
