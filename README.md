# CartWheel Training
Training code for NetVLAD. Re-Implemented with tensorflow. Use of proposed cost-function for recognizing places for loop-closure detection.


## Author
Manohar Kuse <mpkuse@connect.ust.hk> <br/>


## Required Packages
Panda3D - Rendering (only if you use PandaRender.py/PandaRender)<br/>
TensorFlow - Deep learning toolkit (v.1.0+)<br/>
PIL - Image Processing </br/>
skimage - Image Processing <br/>
cv2 - OpenCV <br/>
numpy - Python Math


## Howto train?
Main code : train_netvlad.py. Following will load the `config/A.json` as vital training parameters
and write the output to folder `tf3.logs/A`. You may try other configurations.

It also uses the Pitts250k dataset (should be in `cartwheel_train/data_Akihiko_Torii/Pitssburg`). For info on datasets see next section.

Usage:
```
python train_netvlad.py -t tf3.logs/A/ -f config/A.json
```

## Howto obtain image descriptor?
If you wish to obtain image descriptors for your own images, look at the script `association_map.py`.
You can also obtain the association maps (similar to ones shown in the paper).
You just need trained models and images for which you wish to compute the
descriptors. A few images provided in `sample_images`.

## Training Data
You can request the Pitts250k dataset from : [NetVLAD: CNN architecture for weakly supervised place recognition](http://www.di.ens.fr/willow/research/netvlad/)

It is also possible to train this network with a 3D model using Panda3d rendering engine. Put an issue in this repo if you wish to set it up for yourself. I believe I can help you set it up.
Alternately, see the script `test_render.py` and try getting your panda3d to work. In the
future I will make it easy to train with 3d models (in OBJ format).

Walking videos datatset with/without SLAM data. (work in progress).

Additionally there is also a Google Street view API which you can crawl yourself to generate data.
In the future will provide self collect street view data.

## Trained Model
A few Pre-trained models will be provided for comparison / reference. TODO.

#### K=16
- ResNet6 with triplet-ranking loss.
- ResNet6 with pairwise loss + positive set deviation penalty.
- ResNet6 with pairwise loss
- VGG6 with triplet-ranking loss
- VGG6 with pairwise loss + positive set deviation penalty.

## References
If you use my data/code or if you compare with my results, please do cite. Also cite
the NetVLAD paper whenever appropriate.

- My Paper (Will be added after acceptance)
- Arandjelovic, Relja, et al. "NetVLAD: CNN architecture for weakly supervised place recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.

## Copyright Notice
Released under [MIT license](https://opensource.org/licenses/MIT) unless stated otherwise. The MIT license lets you do anything with the code as long as you provide acknowledgement to me on code use and do not hold me liable for damages if any. Not for commercial use. Contact me
if you wish to use it commercially.
