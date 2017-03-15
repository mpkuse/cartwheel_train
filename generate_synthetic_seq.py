""" Generate Synthetic Sequence

    A renderered sequence. The format is similar to kitti dataset.
    There will be multiple sequences numbered like 00, 01, 02 and so on.
    There are 2 folders viz. `poses` and `sequences`.

    `sequences`
    contains a folder for each sequence containing raw images with file names as
    000000.png, 000001.png, etc.

    `poses`
    contains 1 .txt file per sequence, named 00.txt, 01.txt etc. Each will
    contain 12 number per row. these 12 numbers are the [R|t] 3x4 matrix in
    row major format. The number of lines in this file will be equal to number
    of raw images in itz corresponding folder

    Author  : Manohar Kuse <mpkuse@connect.ust.hk>
    Created : 15th Mar, 2017
"""

import time
import argparse
import code
import os

import cv2
import numpy as np
from PandaRender import TestRenderer
from PathMaker import PathMaker


#
import TerminalColors
tcolor = TerminalColors.bcolors()


# Parse CMD-arg
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--dataset_path", help="Path of the dataset. default : ./other_seqs/altizure_seq/" )
parser.add_argument("-i", "--dataset_id", help="Path of the dataset, default : 00" )
parser.add_argument("-pf", "--path_function_name", help="Name of the function that generates the path. Must be a member of class `PathMaker`. Default : path_bighelix" )
args = parser.parse_args()

if args.dataset_path:
    PARAM_dataset_path = args.dataset_path
else:
    PARAM_dataset_path = './other_seqs/altizure_seq/'

if args.dataset_id:
    PARAM_dataset_id = args.dataset_id
else:
    PARAM_dataset_id = '00'

if args.path_function_name:
    path_function_name = args.path_function_name
else:
    path_function_name = 'path_bighelix'


print tcolor.HEADER, 'PARAM_dataset_path : ', PARAM_dataset_path, tcolor.ENDC
print tcolor.HEADER, 'PARAM_dataset_id   : ', PARAM_dataset_id, tcolor.ENDC
print tcolor.HEADER, 'path_function_name   : ', path_function_name, tcolor.ENDC


# make dir
print 'makedir : '
images_folder_path = PARAM_dataset_path+'/sequences/'+PARAM_dataset_id
os.mkdir( images_folder_path )
print tcolor.OKGREEN, 'OK', tcolor.ENDC


# open poses file
poses_file_name = PARAM_dataset_path+'/poses/%s.txt' %(PARAM_dataset_id)
fp = open( poses_file_name, 'w' )
print 'tcolor.GREEN', 'Open poses file : ',poses_file_name, tcolor.ENDC


task_start = False
task_ended = False


pathGen = eval( 'PathMaker().%s' %(path_function_name) )
app = TestRenderer(  pathGen )
y = None
while y is None:
    im, y = app.step()
    task_start = True

t = 0
while task_ended == False:
    # print app.taskMgr.running
    im, y = app.step()

    if y is None:
        task_ended = True
        break

    # Save im, save y
    image_file_name = images_folder_path+'/%06d.png' %(t)
    cv2.imwrite(image_file_name, cv2.cvtColor(im, cv2.COLOR_RGB2BGR ) )
    print 'Write : ', image_file_name
    fp.write( '1.0 0.0 0.0 %6.4f 0.0 1.0 0.0 %6.4f 0.0 0.0 1.0 %6.4f\n' %(y[0], y[1], y[2]) )
    t = t + 1



fp.close()
print 'Summary of Writing'
print t, 'images in ', images_folder_path
print 'poses file : ', poses_file_name
