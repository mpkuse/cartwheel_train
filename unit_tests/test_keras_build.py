""" Build a network. test code  """

import keras

input_img = keras.layers.Input( shape=(640, 480, 3 ) )

if False:
    base_model = keras.applications.vgg19.VGG19(weights=None, include_top=False, input_tensor=input_img)
    keras.utils.plot_model( base_model, to_file='VGG19.png', show_shapes=True )


if False:
    base_model = keras.applications.vgg16.VGG16(weights=None, include_top=False, input_tensor=input_img)
    keras.utils.plot_model( base_model, to_file='VGG16.png', show_shapes=True )


if False:
    base_model = keras.applications.resnet50.ResNet50(weights=None, include_top=False, input_tensor=input_img)
    keras.utils.plot_model( base_model, to_file='ResNet50.png', show_shapes=True )



if False:
    base_model = keras.applications.inception_v3.InceptionV3(weights=None, include_top=False, input_tensor=input_img)
    keras.utils.plot_model( base_model, to_file='InceptionV3.png', show_shapes=True )

if False:
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(weights=None, include_top=False, input_tensor=input_img)
    keras.utils.plot_model( base_model, to_file='InceptionResNetV2.png', show_shapes=True )


if False:
    base_model = keras.applications.mobilenet.MobileNet(weights=None, include_top=False, input_tensor=input_img)
    keras.utils.plot_model( base_model, to_file='MobileNet.png', show_shapes=True )


if True:
    base_model = keras.applications.mobilenet_v2.MobileNetV2(weights=None, include_top=False, input_tensor=input_img)
    keras.utils.plot_model( base_model, to_file='MobileNetV2.png', show_shapes=True )

base_model.summary()
