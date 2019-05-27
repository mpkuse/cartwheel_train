from keras import backend as K
from keras.engine.topology import Layer
import keras
import code
import numpy as np

import cv2
import code

from imgaug import augmenters as iaa
import imgaug as ia


#--------------------------------------------------------------------------------
# Data
#--------------------------------------------------------------------------------
# TODO : removal
def dataload_( n_tokyoTimeMachine, n_Pitssburg, nP, nN ):
    D = []
    if n_tokyoTimeMachine > 0 :
        TTM_BASE = '/Bulk_Data/data_Akihiko_Torii/Tokyo_TM/tokyoTimeMachine/' #Path of Tokyo_TM
        pr = TimeMachineRender( TTM_BASE )
        print 'tokyoTimeMachine:: nP=', nP, '\tnN=', nN
        for s in range(n_tokyoTimeMachine):
            a,_ = pr.step(nP=nP, nN=nN, return_gray=False, resize=(320,240), apply_distortions=False, ENABLE_IMSHOW=False)
            if s%100 == 0:
                print 'get a sample Tokyo_TM #%d of %d\t' %(s, n_tokyoTimeMachine),
                print a.shape
            D.append( a )

    if n_Pitssburg > 0 :
        PTS_BASE = '/Bulk_Data/data_Akihiko_Torii/Pitssburg/'
        pr = PittsburgRenderer( PTS_BASE )
        print 'Pitssburg nP=', nP, '\tnN=', nN
        for s in range(n_Pitssburg):
            a,_ = pr.step(nP=nP, nN=nN, return_gray=False, resize=(240,320), apply_distortions=False, ENABLE_IMSHOW=False)
            if s %100 == 0:
                print 'get a sample Pitssburg #%d of %d\t' %(s, n_Pitssburg),
                print a.shape
            D.append( a )

    return D


def do_augmentation( D ):
    """ D : Nx(n+p+1)xHxWx3. Return N1x(n+p+1)xHxWx3 """

    n_samples = D.shape[0]
    n_images_per_sample = D.shape[1]

    im_rows = D.shape[2]
    im_cols = D.shape[3]
    im_chnl = D.shape[4]

    E = D.reshape( n_samples*n_images_per_sample,  im_rows,im_cols,im_chnl  )


    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Very basic
    if True:
        seq = iaa.Sequential([
            sometimes( iaa.Crop(px=(0, 50)) ), # crop images from each side by 0 to 16px (randomly chosen)
            # iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
            sometimes( iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
                ) )
        ])
        seq_vbasic = seq

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.

    # Typical
    if True:
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True) # apply augmenters in random order
        # seq = sometimes( seq )
        seq_typical = seq

    # Heavy
    if True:
        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.2), # horizontally flip 20% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            #iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.OneOf([
                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(
                                exponent=(-4, 0),
                                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                second=iaa.ContrastNormalization((0.5, 2.0))
                            )
                        ]),
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
        seq_heavy = seq

    print 'Add data'
    L = [E]
    print 'seq_vbasic'
    L.append( seq_vbasic.augment_images(E) )
    print 'seq_typical'
    L.append( seq_typical.augment_images(E) )
    print 'seq_typical'
    L.append( seq_typical.augment_images(E) )
    print 'seq_heavy'
    L.append( seq_heavy.augment_images(E) )

    G = [ l.reshape(n_samples, n_images_per_sample, im_rows,im_cols,im_chnl) for l in L ]
    G = np.concatenate( G )
    print 'Input.shape ', D.shape, '\tOutput.shape ', G.shape
    return G

    # for j in range(n_times):
    #     images_aug = seq.augment_images(E)
    #     # L.append( images_aug.reshape( n_samples, n_images_per_sample, im_rows,im_cols,im_chnl  ) )
    #     L.append( images_aug )

    # code.interact( local=locals() )
    return L



def do_typical_data_aug( D ):
    """ D : Nx(n+p+1)xHxWx3. Return N1x(n+p+1)xHxWx3 """
    D = np.array( D )
    assert( len(D.shape) == 5 )
    print '[do_typical_data_aug]', 'D.shape=', D.shape

    n_samples = D.shape[0]
    n_images_per_sample = D.shape[1]

    im_rows = D.shape[2]
    im_cols = D.shape[3]
    im_chnl = D.shape[4]

    E = D.reshape( n_samples*n_images_per_sample,  im_rows,im_cols,im_chnl  )


    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes_2 = lambda aug: iaa.Sometimes(0.2, aug)


    seq = iaa.Sequential( [
        #iaa.Fliplr(0.5), # horizontal flips
        #iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    D = seq.augment_images(E)
    D = D.reshape(n_samples, n_images_per_sample, im_rows,im_cols,im_chnl)
    print '[do_typical_data_aug] Done...!', 'D.shape=', D.shape

    return D
