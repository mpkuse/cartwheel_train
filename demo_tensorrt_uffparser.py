# To check if .uff file can be loaded back

import tensorrt as trt

TRT_LOGGER = trt.Logger( trt.Logger.WARNING)
# LOG_DIR = 'models.keras/June2019/centeredinput-m1to1-240x320x3__mobilenet-conv_pw_6_relu__K16__allpairloss/'
LOG_DIR = 'models.keras/June2019/centeredinput-m1to1-240x320x3__mobilenetv2-block_9_add__K16__allpairloss/'
uff_fname = 'output_nvinfer.uff'

with trt.Builder( TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    # Set inputs and outputs correctly as per the model-uff
    parser.register_input("input_1", (3,240,320) )
    # parser.register_output( "conv_pw_5_relu/Relu6" )
    parser.register_output( "net_vlad_layer_1/l2_normalize_1" )
    parser.parse( LOG_DIR+'/'+uff_fname, network )
    pass


# TODO
# you need pycuda for this
# 1. Load Image to GPU with cudamemcpy HtoD.
# 2. Execute
# 3. cudamemcpy DtoH
