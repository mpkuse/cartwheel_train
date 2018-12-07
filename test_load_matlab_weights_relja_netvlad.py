# Get the trained model from Relja https://www.di.ens.fr/willow/research/netvlad/ (best model)
# and translate into keras
# Use the following matlab script to translate the matlab model to keras.
# ``` computeRepresentation.m
# clear all;
# clc;
# setup;
# netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
# paths= localPaths();
# load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
# net= relja_simplenn_tidy(net);
#
# im= vl_imreadjpeg({which('football.jpg')});
# im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
# feats= computeRepresentation(net, im); % add `'useGPU', false` if you want to use the CPU
#
# save( 'outfolder/im.mat', 'im' )
# save( 'outfolder/feats.mat', 'feats' )
#
# for i=1:size(net.layers,2)
#     %mat = cell2mat(net.layers{i});
#     layer = net.layers{i};
#
#
#     ty = layer.type; %getfield( mat, 'type' ) ;
#     name = layer.name; %getfield( mat, 'name' ) ;
#
#     if any(strcmp(fieldnames(layer), 'weights')) == false
#         display( 'no weight' );
#         continue;
#     end
#
#     s1 = size(net.layers{i}.weights, 1);
#     s2 = size(net.layers{i}.weights, 2);
#
#     display( sprintf('%d: %s : %s; s=%dx%d', i, ty, name, s1, s2 ) );
#
#
#     if s1>0
#         %display( 'yes' );
#         for k=1:s2
#             display( sprintf( "\t%d", k ) );
#             the_mat = net.layers{i}.weights{1,k};
#             size(net.layers{i}.weights{1,k})
#
#             fname = sprintf( 'outfolder/%s_%d.mat', name, k )
#             save( fname, 'the_mat' );
#         end
#         %kerrnel = net.layers{i}.weights{1,1};
#         %bias = net.layers{i}.weights{1,2};
#     end
#
# end
# ```

import keras
import code
import time
import numpy as np

import scipy.io
import glob
import os.path

# CustomNets
from CustomNets import NetVLADLayer
from CustomNets import make_from_vgg16, make_from_mobilenet

def make_keras_model_org_netvlad(im_rows = 240, im_cols = 320, im_chnls = 3 ):
    # im_rows = 240
    # im_cols = 320
    # im_chnls = 3
    input_img = keras.layers.Input( batch_shape=(1,im_rows, im_cols, im_chnls ) )
    cnn = make_from_vgg16( input_img, weights=None, layer_name='block5_pool' )
    out, out_amap = NetVLADLayer(num_clusters = 64)( cnn )
    model = keras.models.Model( inputs=input_img, outputs=out )
    return model


def get_translation_map():
    k = 0
    # block1_conv1 (Conv2D)        (1, 240, 320, 64)         1792
    # block1_conv2 (Conv2D)        (1, 240, 320, 64)         36928

    # block2_conv1 (Conv2D)        (1, 120, 160, 128)        73856
    # block2_conv2 (Conv2D)        (1, 120, 160, 128)        147584

    # block3_conv1 (Conv2D)        (1, 60, 80, 256)          295168
    # block3_conv2 (Conv2D)        (1, 60, 80, 256)          590080
    # block3_conv3 (Conv2D)        (1, 60, 80, 256)          590080

    # block4_conv1 (Conv2D)        (1, 30, 40, 512)          1180160
    # block4_conv2 (Conv2D)        (1, 30, 40, 512)          2359808
    # block4_conv3 (Conv2D)        (1, 30, 40, 512)          2359808

    # block5_conv1 (Conv2D)        (1, 15, 20, 512)          2359808
    # block5_conv2 (Conv2D)        (1, 15, 20, 512)          2359808
    # block5_conv3 (Conv2D)        (1, 15, 20, 512)          2359808

    # net_vlad_layer_1 (NetVLADLay [(1, 32768), (1, 7, 10)]  65600
    M = {}
    M['conv1_1'] = 'block1_conv1'
    M['conv1_2'] = 'block1_conv2'
    M['conv2_1'] = 'block2_conv1' #####
    M['conv2_2'] = 'block2_conv2'
    M['conv3_1'] = 'block3_conv1' ####
    M['conv3_2'] = 'block3_conv2'
    M['conv3_3'] = 'block3_conv3'
    M['conv4_1'] = 'block4_conv1' ####
    M['conv4_2'] = 'block4_conv2'
    M['conv4_3'] = 'block4_conv3'
    M['conv5_1'] = 'block5_conv1' ####
    M['conv5_2'] = 'block5_conv2'
    M['conv5_3'] = 'block5_conv3'

    N = {}
    N['block1_conv1'] = 'conv1_1'
    N['block1_conv2'] = 'conv1_2'
    N['block2_conv1'] = 'conv2_1'
    N['block2_conv2'] = 'conv2_2'
    N['block3_conv1'] = 'conv3_1'
    N['block3_conv2'] = 'conv3_2'
    N['block3_conv3'] = 'conv3_3'
    N['block4_conv1'] = 'conv4_1'
    N['block4_conv2'] = 'conv4_2'
    N['block4_conv3'] = 'conv4_3'
    N['block5_conv1'] = 'conv5_1'
    N['block5_conv2'] = 'conv5_2'
    N['block5_conv3'] = 'conv5_3'

    return M, N





# Load matlab weights
DATA_DIR = './relja_matlab_weight.dump/'


# Load input and outpt sample
input_im = scipy.io.loadmat( DATA_DIR+'/im.mat' )['im']
avg_image = [122.6778, 116.6522, 103.9997]
input_im[:,:,0]= input_im[:,:,0] - avg_image[0]#- np.mean(input_im[:,:,0]) #- net.meta.normalization.averageImage(1,1,1);
input_im[:,:,1]= input_im[:,:,1] - avg_image[1]#- np.mean(input_im[:,:,1])#- net.meta.normalization.averageImage(1,1,2);
input_im[:,:,2]= input_im[:,:,2] - avg_image[2]#- np.mean(input_im[:,:,2])#- net.meta.normalization.averageImage(1,1,3);

out_feat = np.transpose( scipy.io.loadmat( DATA_DIR+'/feats.mat' )['feats'] ) # after transpose : 1x4096




# Make a keras model
model = make_keras_model_org_netvlad( input_im.shape[0], input_im.shape[1] )
model.summary()

if True: # Load Weights from .mat files
    matlab_to_keras, keras_to_matlab = get_translation_map()

    # Load Conv weights
    for matlab_layer in matlab_to_keras.keys():
        print '---'
        kernel_fname = DATA_DIR+'/'+matlab_layer+'_1.mat'
        bias_fname   = DATA_DIR+'/'+matlab_layer+'_2.mat'
        print 'fname=', kernel_fname, bias_fname
        kernel = scipy.io.loadmat( kernel_fname )['the_mat']
        bias = scipy.io.loadmat( bias_fname )['the_mat']
        print 'matlab.kernel=', kernel.shape, '\tmatlab.bias=', bias.shape

        keras_layer = model.get_layer( matlab_to_keras[matlab_layer] )
        print keras_layer.name
        keras_layer.set_weights( [kernel,bias[0]] )


    # Load NetVLAD weights
    f1_fname = DATA_DIR + '/' + 'vlad:core_1' + '.mat'
    f2_fname = DATA_DIR + '/' + 'vlad:core_2' + '.mat'
    print 'fname_vlad_weights = ', f1_fname, f2_fname
    f1 = scipy.io.loadmat( f1_fname )['the_mat']
    f2 = scipy.io.loadmat( f2_fname )['the_mat']
    print 'f1.shape=', f1.shape, '\t', 'f2.shape=', f2.shape

    keras_netvlad_layer = model.get_layer( 'net_vlad_layer_1' )
    keras_netvlad_layer.set_weights( [f1, np.zeros( (1,1,64) ), np.expand_dims(f2,0) ] )


    # save keras model
    print 'Save Keras Model : ', DATA_DIR+'/matlab_model.keras'
    model.save( DATA_DIR+'/matlab_model.keras' )
else: # Load weights from keras
    model.load_weights( DATA_DIR+'/matlab_model.keras' )

# Load WPCA Matrix
WPCA_M_fname = DATA_DIR+'/WPCA_1.mat'
WPCA_b_fname = DATA_DIR+'/WPCA_2.mat'
print 'Load WPCA matrix: ', WPCA_M_fname, WPCA_b_fname
WPCA_M = scipy.io.loadmat( WPCA_M_fname )['the_mat'] # 1x1x32768x4096
WPCA_b = scipy.io.loadmat( WPCA_b_fname )['the_mat'] # 4096x1
WPCA_M = WPCA_M[0,0]          # 32768x4096
WPCA_b = np.transpose(WPCA_b) #1x4096
print 'WPCA_M.shape=', WPCA_M.shape
print 'WPCA_b.shape=', WPCA_b.shape


# model.predict
my_out_im_descp = model.predict( np.expand_dims( input_im , 0 ) )
my_out_im_descp_reduced = np.matmul( my_out_im_descp, WPCA_M ) + WPCA_b
my_out_im_descp_reduced /= np.linalg.norm( my_out_im_descp_reduced )

print  np.corrcoef(my_out_im_descp_reduced[0], out_feat[0] )

# code.interact( local=locals() )
# quit()
