# CartWheel Training
Cast the place recognition problem as a classification problem. I divide the whole trainable area as grid. With deep resnet make a classification problem. Images are rendered from the 3D model. 

## Author
Manohar Kuse <mpkuse@connect.ust.hk> <br/>
6th Jan, 2017

## Required Packages
### You might need these
Panda3D - Rendering <br/>
TensorFlow - Deep learning toolkit <br/>
PIL - Image Processing </br/>
skimage - Image Processing <br/>
cv2 - OpenCV <br/>
annoy - Approximate Nearest Neighbour (ANN) <br/>

### You probably already have these
argparse<br/>
code<br/>
copy<br/>
cv2<br/>
matplotlib.pyplot<br/>
numpy<br/>
pickle<br/>
Queue<br/>
time<br/>


### Note
You will need to have im_batch.pickle and label_batch.pickle which is used in the renderers in case the queue is not fully filled.


### 
likehood_ratio_test, thresh= 0.76
Pr(same place) = 0.3333
Pr(diff place) = 0.6667
Pr(pred to be same) = 0.3073
Pr(pred to be diff) = 0.6927
Pr(pred to b same/same) = 0.8770
Pr(pred to b diff/same) = 0.1230
Pr(pred to b same/diff) = 0.0225
Pr(pred to b diff/diff) = 0.9775
Pr(same/pred to b same) = 0.9512
Pr(same/pred to b diff) = 0.1334
Pr(diff/pred to b same) = 0.0217
Pr(diff/pred to b diff) = 0.9408

