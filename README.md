# CartWheel Training
Training code for NetVLAD. Re-Implemented with tensorflow. Use of proposed cost-function for recognizing places for loop-closure detection. 


## Author
Manohar Kuse <mpkuse@connect.ust.hk> <br/>


## Required Packages
### You might need these
Panda3D - Rendering (only if you use PandaRender.py/PandaRender)<br/>
TensorFlow - Deep learning toolkit (v.1.0+)<br/>
PIL - Image Processing </br/>
skimage - Image Processing <br/>
cv2 - OpenCV <br/>
numpy - Python Math


### Howto use
Main code : train_netvlad.py. Following will load the `config/A.json` as vital training parameters and
write the output to folder `tf3.logs/A`.

It also uses the Pitts250k dataset. For info on datasets see next section.

Usage:
```
python train_netvlad.py -t tf3.logs/A/ -f config/A.json
```

### Data
You can request the Pitts250k dataset from : [NetVLAD: CNN architecture for weakly supervised place recognition](http://www.di.ens.fr/willow/research/netvlad/)

It is also possible to train this network with a 3D model using Panda3d rendering engine. Put an issue in this repo if you wish to set it up for yourself. I believe I can help you set it up. 

Additionally there is also a Google Street view API which you can crawl yourself to generate data. 


