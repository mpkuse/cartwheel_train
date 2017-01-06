""" The training-render file. (class TrainRenderer)
        Update: This class is render only class. It does not take the mainloop()
        control. Basically need to call function step(). Works with
        `train_tf_decop.p`

        Defines a rendering class. Defines a spinTask (panda3d) which basicalyl
        renders 16-cameras at a time and sets them into a CPU-queue.

        New:
        Training with TensorFlow (Now done externally, )
        #TODO : Cleanup unwanted training functions

        OLD Approach: (Now defunct)
        After setting it into queue, a class function call_caffe() is called
        which transfroms the data and sets it into caffe, does 1 iteration
        and returns back to spinTask. Caffe prototxt and solver files loaded in
        class constructor.


"""

# Panda3d
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.stdpy.threading2 import Thread

# Usual Math and Image Processing
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import caffe
# import tensorflow as tf



# Other System Libs
import os
import argparse
import Queue
import copy
import time
import code
import pickle


# Custom-Misc
import TerminalColors
import CubeMaker
import censusTransform as ct
import Noise
# import PlutoFlow_noBN as puf
# import PlutoFlow as puf

class TrainRenderer(ShowBase):
    renderIndx=0


    # Given a square matrix, substract mean and divide std dev
    def zNormalized(self, M ):
        M_mean = np.mean(M) # scalar
        M_std = np.std(M)
        if M_std < 0.0001 :
            return M

        M_statSquash = (M - M_mean)/(M_std+.0001)
        return M_statSquash

    # Basic Mesh & Camera Setup
    def loadAllTextures(self, mesh, basePath, silent=True):
        """ Loads texture files for a mesh """
        c = 0
        for child in mesh.getChildren():
            submesh_name = child.get_name()
            submesh_texture = basePath + submesh_name[:-5] + 'tex0.jpg'
            child.setTexture( self.loader.loadTexture(submesh_texture) )

            if silent == False:
                print 'Loading texture file : ', submesh_texture
            c = c + 1

        print self.tcolor.OKGREEN, "Loaded ", c, "textures", self.tcolor.ENDC
    def setupMesh(self):
        """ Loads the .obj files. Will load mesh sub-divisions separately """

        print 'Attempt Loading Mesh VErtices, FAces'
        self.cyt = self.loader.loadModel( 'model_l/l6/level_6_0_0.obj' )
        self.cyt2 = self.loader.loadModel( 'model_l/l6/level_6_128_0.obj' )

        self.low_res = self.loader.loadModel( 'model_l/l3/level_3_0_0.obj' )
        print self.tcolor.OKGREEN, 'Done Loading Vertices', self.tcolor.ENDC

        print 'Attempt Loading Textures'
        self.loadAllTextures( self.cyt, 'model_l/l6/')
        self.loadAllTextures( self.cyt2, 'model_l/l6/')
        self.loadAllTextures( self.low_res, 'model_l/l3/')
        print self.tcolor.OKGREEN, 'Done Loading Textures', self.tcolor.ENDC

    def positionMesh(self):
        """ WIll have to manually adjust this for ur mesh. I position the
        center where I fly my drone and oriented in ENU (East-north-up)
        cords for easy alignment of GPS and my cordinates. If your model
        is not metric scale will have to adjust for that too"""

        self.cyt.setPos( 140,-450, 150 )
        self.cyt2.setPos( 140,-450, 150 )
        self.low_res.setPos( 140,-450, 150 )
        self.cyt.setHpr( 198, -90, 0 )
        self.cyt2.setHpr( 198, -90, 0 )
        self.low_res.setHpr( 198, -90, 0 )
        self.cyt.setScale(172)
        self.cyt2.setScale(172)
        self.low_res.setScale(172)

    def customCamera(self, nameIndx):
        lens = self.camLens
        lens.setFov(83)
        print 'self.customCamera : Set FOV at 83'
        my_cam = Camera("cam"+nameIndx, lens)
        my_camera = self.scene0.attachNewNode(my_cam)
        # my_camera = self.render.attachNewNode(my_cam)
        my_camera.setName("camera"+nameIndx)
        return my_camera
    def customDisplayRegion(self, rows, cols):
        rSize = 1.0 / rows
        cSize = 1.0 / cols

        dr_list = []
        for i in range(0,rows):
            for j in range(0,cols):
                # print i*rSize, (i+1)*rSize, j*cSize, (j+1)*cSize
                dr_i = self.win2.makeDisplayRegion(i*rSize, (i+1)*rSize, j*cSize, (j+1)*cSize)
                dr_i.setSort(-5)
                dr_list.append( dr_i )
        return dr_list

    def learning_iteration(self):
        """Does 1 iteration.
            Basically retrives the rendered image and pose from queue. Preprocess/Purturb it
            and feed it into caffe for training.
        """
        # print 'DoCaffeTrainng'
        startTime = time.time()
        # print 'q_imStack.qsize() : ', self.q_imStack.qsize()
        # print 'q_labelStack.qsize() : ', self.q_labelStack.qsize()


        # if too few items in queue do not proceed with iterations
        if self.q_imStack.qsize() < 16*5:
            return None



        batchsize = 20
        im_batch = np.zeros((batchsize,240,320,3))
        label_batch = np.zeros((batchsize,4))
        # print 'batchsize', batchsize
        # print "self.solver.net.blobs['data'].data", self.solver.net.blobs['data'].data.shape
        # print "self.solver.net.blobs['label_x'].data",self.solver.net.blobs['label_x'].data.shape
        for i in range(batchsize):
            im = self.q_imStack.get() #240x320x3 RGB
            y = self.q_labelStack.get()

            im_noisy = Noise.noisy( 'gauss', im )
            im_gry = np.mean( im_noisy, axis=2)


            # print im.shape
            # itr_indx = TrainRenderer.renderIndx
            # cv2.imwrite( 'dump/'+str(itr_indx)+'_'+str(i)+'.png', cv2.cvtColor( im.astype('uint8'), cv2.COLOR_BGR2RGB ) )

            #remember to z-normalize
            im_batch[i,:,:,0] = self.zNormalized( im[:,:,0] )
            im_batch[i,:,:,1] = self.zNormalized( im[:,:,1] )
            im_batch[i,:,:,2] = self.zNormalized( im[:,:,2] )
            label_batch[i,0] = y[0]
            label_batch[i,1] = y[1]
            label_batch[i,2] = y[2]
            label_batch[i,3] = y[3]


            # cencusTR = ct.censusTransform( im_gry.astype('uint8') )
            # edges_out = cv2.Canny(cv2.blur(im_gry.astype('uint8'),(3,3)),100,200)

        # code.interact(local=locals())

        lr = self.get_learning_rate( self.tensorflow_iteration )
        feed = {\
        self.tf_tower_ph_x[0]:im_batch[0:10,:,:,:],\
        self.tf_tower_ph_label_x[0]:label_batch[0:10,0:1], \
        self.tf_tower_ph_label_y[0]:label_batch[0:10,1:2], \
        self.tf_tower_ph_label_z[0]:label_batch[0:10,2:3], \
        self.tf_tower_ph_label_yaw[0]:label_batch[0:10,3:4], \
        self.tf_tower_ph_x[1]:im_batch[10:20,:,:,:],\
        self.tf_tower_ph_label_x[1]:label_batch[10:20,0:1], \
        self.tf_tower_ph_label_y[1]:label_batch[10:20,1:2], \
        self.tf_tower_ph_label_z[1]:label_batch[10:20,2:3], \
        self.tf_tower_ph_label_yaw[1]:label_batch[10:20,3:4], \
        self.tf_learning_rate:lr}

        # _,aa,bb = self.tensorflow_session.run( [\
        #                 self.tensorflow_apply_grad,\
        #                 self.tf_tower_cost[0],\
        #                 self.tf_tower_cost[1],\
        #                 #self.tensorflow_summary_op \
        #                 ], \
        #                 feed_dict=feed )
        #
        # print '[%4d] : cost=%0.4f,%0.4f ; time=%0.4f ms' %(self.tensorflow_iteration, aa, bb, (time.time() - startTime)*1000.)

        pp,qq = self.tensorflow_session.run( [self.tf_learning_rate,self.tf_tower_cost[0] ], \
                            feed_dict={self.tf_learning_rate:lr, \
                                    self.tf_tower_ph_x[0]:im_batch[0:10,:,:,:],\
                                            self.tf_tower_ph_label_x[0]:label_batch[0:10,0:1], \
                                            self.tf_tower_ph_label_y[0]:label_batch[0:10,1:2], \
                                            self.tf_tower_ph_label_z[0]:label_batch[0:10,2:3], \
                                            self.tf_tower_ph_label_yaw[0]:label_batch[0:10,3:4], \
                                    self.tf_tower_ph_x[1]:im_batch[10:20,:,:,:],\
                            } )
        print qq

        # # Write Summary for TensorBoard
        # if self.tensorflow_iteration % self.PARAM_WRITE_SUMMARY_EVERY == 0 and self.tensorflow_iteration > 0:
        #     print 'write_summary()'
        #     self.tensorflow_summary_writer.add_summary( ss, self.tensorflow_iteration )

        # Snapshot model
        if self.tensorflow_iteration % self.PARAM_WRITE_TF_MODEL_EVERY == 0 and self.tensorflow_iteration > 0:
            sess = self.tensorflow_session
            pth = self.PARAM_MODEL_SAVE_PREFIX
            step = self.tensorflow_iteration
            save_path = self.tensorflow_saver.save( sess, pth, global_step=step )
            print 'snapshot model()', save_path


        # Try testing every 100 iterations
        # if self.tensorflow_iteration % 100 == 0 and self.tensorflow_iteration > 0:
            # self.do_test_evaluation(100)





        self.tensorflow_iteration = self.tensorflow_iteration + 1

    def do_test_evaluation( self, batchsize=100 ):
        """ Does inference using self.tf_infer_op. Note that this function will do dequeue.
        Warning: This might cause a race condition if too many are extracted from the queue
        """

        # Make a batch - Dequeue and batch (similar to training)
        im_batch = np.zeros((batchsize,240,320,3))
        label_batch = np.zeros((batchsize,4))
        # print 'batchsize', batchsize
        # print "self.solver.net.blobs['data'].data", self.solver.net.blobs['data'].data.shape
        # print "self.solver.net.blobs['label_x'].data",self.solver.net.blobs['label_x'].data.shape
        for i in range(batchsize):
            im = self.q_imStack.get() #240x320x3 RGB
            y = self.q_labelStack.get()


            im_batch[i,:,:,0] = self.zNormalized( im[:,:,0] )
            im_batch[i,:,:,1] = self.zNormalized( im[:,:,1] )
            im_batch[i,:,:,2] = self.zNormalized( im[:,:,2] )
            label_batch[i,0] = y[0]
            label_batch[i,1] = y[1]
            label_batch[i,2] = y[2]
            label_batch[i,3] = y[3]

        # session.run() for inference
        aa_out = self.tensorflow_session.run( [self.tf_infer_op], feed_dict={ self.tf_x:im_batch } ) #1x4x12x1
        aa_out = aa_out[0]


        # Print side by side
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].plot( aa_out[0], 'r' )
        axarr[0, 0].plot( label_batch[:,0], 'b' )
        axarr[0, 0].set_title('X')

        axarr[0, 1].plot( aa_out[1], 'r' )
        axarr[0, 1].set_title('Y')

        axarr[1, 0].plot( aa_out[2], 'r' )
        axarr[0, 1].plot( label_batch[:,1], 'b' )
        axarr[1, 0].plot( label_batch[:,2], 'b' )
        axarr[1, 0].set_title('Z')

        axarr[1, 1].plot( aa_out[3], 'r' )
        axarr[1, 1].plot( label_batch[:,3], 'b' )
        axarr[1, 1].set_title('Yaw')


        plt.savefig('tf.logs/foo'+str(self.tensorflow_iteration)+'.png')
        # plt.show()


        # code.interact(local=locals())







    def monte_carlo_sample(self):
        """ Gives a random 6-dof pose. Need to set params manually here.
                X,Y,Z,  Yaw(abt Z-axis), Pitch(abt X-axis), Roll(abt Y-axis) """


        # mc_X_min etc are set in constructor
        X = np.random.uniform(self.mc_X_min,self.mc_X_max)
        Y = np.random.uniform(self.mc_Y_min,self.mc_Y_max)
        Z = np.random.uniform(self.mc_Z_min,self.mc_Z_max)

        yaw = np.random.uniform( self.mc_yaw_min, self.mc_yaw_max)
        roll = np.random.uniform( self.mc_roll_min, self.mc_roll_max)
        pitch = np.random.uniform( self.mc_pitch_min, self.mc_pitch_max)

        return X,Y,Z, yaw,roll,pitch

    # Annotation-helpers for self.render
    def putBoxes(self,X,Y,Z,r=1.,g=0.,b=0., scale=1.0):
        cube_x = CubeMaker.CubeMaker().generate()
        cube_x.setColor(r,g,b)
        cube_x.setScale(scale)
        cube_x.reparentTo(self.render)
        cube_x.setPos(X,Y,Z)

    def putTrainingBox(self,task):
        cube = CubeMaker.CubeMaker().generate()

        cube.setTransparency(TransparencyAttrib.MAlpha)
        cube.setAlphaScale(0.5)

        # cube.setScale(10)
        # mc_X_min etc are set in constructor
        sx = 0.5 * (self.mc_X_max - self.mc_X_min)
        sy = 0.5 * (self.mc_Y_max - self.mc_Y_min)
        sz = 0.5 * (self.mc_Z_max - self.mc_Z_min)

        ax = 0.5 * (self.mc_X_max + self.mc_X_min)
        ay = 0.5 * (self.mc_Y_max + self.mc_Y_min)
        az = 0.5 * (self.mc_Z_max + self.mc_Z_min)

        cube.setSx(sx)
        cube.setSy(sy)
        cube.setSz(sz)
        cube.reparentTo(self.render)
        cube.setPos(ax,ay,az)



    def putAxesTask(self,task):
        if (task.frame / 10) % 2 == 0:
            cube_x = CubeMaker.CubeMaker().generate()
            cube_x.setColor(1.0,0.0,0.0)
            cube_x.setScale(1)
            cube_x.reparentTo(self.render)
            cube_x.setPos(task.frame,0,0)

            cube_y = CubeMaker.CubeMaker().generate()
            cube_y.setColor(0.0,1.0,0.0)
            cube_y.setScale(1)
            cube_y.reparentTo(self.render)
            cube_y.setPos(0,task.frame,0)

            cube_z = CubeMaker.CubeMaker().generate()
            cube_z.setColor(0.0,0.0,1.0)
            cube_z.setScale(1)
            cube_z.reparentTo(self.render)
            cube_z.setPos(0,0,task.frame)
        if task.time > 25:
            return None
        return task.cont


    # Render-n-Learn task
    def renderNlearnTask(self, task):
        if task.time < 2: #do not do anything for 1st 2 sec
            return task.cont


        # print randX, randY, randZ

        #
        # set pose in each camera
        # Note: The texture is grided images in a col-major format
        poses = np.zeros( (len(self.cameraList), 4), dtype='float32' )
        for i in range(len(self.cameraList)):
            randX,randY, randZ, randYaw, randPitch, randRoll = self.monte_carlo_sample()
            # if i<4 :
            #     randX = (i) * 30
            # else:
            #     randX = 0
            #
            # randY = 0#task.frame
            # randZ = 80
            # randYaw = 0
            # randPitch = 0
            # randRoll = 0


            self.cameraList[i].setPos(randX,randY,randZ)
            self.cameraList[i].setHpr(randYaw,-90+randPitch,0+randRoll)

            poses[i,0] = randX
            poses[i,1] = randY
            poses[i,2] = randZ
            poses[i,3] = randYaw

        #     self.putBoxes(randX,randY,randZ, scale=0.3)
        #
        # if task.frame < 100:
        #     return task.cont
        # else:
        #     return None



        # make note of the poses just set as this will take effect next
        if TrainRenderer.renderIndx == 0:
            TrainRenderer.renderIndx = TrainRenderer.renderIndx + 1
            self.prevPoses = poses
            return task.cont



        #
        # Retrive Rendered Data
        tex = self.win2.getScreenshot()
        A = np.array(tex.getRamImageAs("RGB")).reshape(960,1280,3)
        # A = np.zeros((960,1280,3))
        # A_bgr =  cv2.cvtColor(A.astype('uint8'),cv2.COLOR_RGB2BGR)
        # cv2.imwrite( str(TrainRenderer.renderIndx)+'.png', A_bgr.astype('uint8') )
        # myTexture = self.win2.getTexture()
        # print myTexture

        # retrive poses from prev render
        texPoses = self.prevPoses

        #
        # Cut rendered data into individual image. Note rendered data will be 4X4 grid of images
        #960 rows and 1280 cols (4x4 image-grid)
        nRows = 240
        nCols = 320
        # Iterate over the rendered texture in a col-major format
        c=0
        if self.q_imStack.qsize() < 150:
            for j in range(4): #j is for cols-indx
                for i in range(4): #i is for rows-indx
                    #print i*nRows, j*nCols, (i+1)*nRows, (j+1)*nCols
                    im = A[i*nRows:(i+1)*nRows,j*nCols:(j+1)*nCols,:]
                    #imX = im.astype('float32')/255. - .5 # TODO: have a mean image
                    #imX = (im.astype('float32') - 128.0) /128.
                    imX = im.astype('float32')  #- self.meanImage

                    # Put imX into the queue
                    # do not queue up if queue size begin to exceed 150


                    self.q_imStack.put( imX )
                    self.q_labelStack.put( texPoses[c,:] )


                    # fname = '__'+str(poses[c,0]) + '_' + str(poses[c,1]) + '_' + str(poses[c,2]) + '_' + str(poses[c,3]) + '_'
                    # cv2.imwrite( str(TrainRenderer.renderIndx)+'__'+str(i)+str(j)+fname+'.png', imX.astype('uint8') )

                    c = c + 1
        else:
            print 'q_imStack.qsize() > 150. Queue is filled, not retriving the rendered data'



        #
        # Call caffe iteration (reads from q_imStack and q_labelStack)
        #       Possibly upgrade to TensorFlow
        # self.learning_iteration()



        # if( TrainRenderer.renderIndx > 50 ):
        #     return None

        #
        # Prep for Next Iteration
        TrainRenderer.renderIndx = TrainRenderer.renderIndx + 1
        self.prevPoses = poses



        return task.cont


    def step(self, batchsize):
        """ One rendering.
        This function needs to be called from outside in a loop for continous rendering
        Returns 2 variables. One im_batch and another label
        """

        # ltimes = int( batchsize/16 ) + 1
        # print 'Render ', ltimes, 'times'
        # for x in range(ltimes):
        # Note: 2 renders sometime fails. Donno exactly what happens :'(
        # Instead I do app.taskMgr.step() in the main() instead, once and 1 time here. This seem to work OK
        # self.taskMgr.step()
        # Thread.sleep(0.1)

        self.taskMgr.step()

        # print 'Queues Status (imStack=%d,labelStack=%d)' %(self.q_imStack.qsize(), self.q_labelStack.qsize())

        # TODO: Check validity of batchsize. Also avoid hard coding the thresh for not retriving from queue.

        im_batch = np.zeros((batchsize,240,320,3))
        label_batch = np.zeros((batchsize,4))

        # assert self.q_imStack.qsize() > 16*5
        if self.q_imStack.qsize() >= 16*5:

            # get a batch out
            for i in range(batchsize):
                im = self.q_imStack.get() #240x320x3 RGB
                y = self.q_labelStack.get()
                # print 'retrive', i


                #remember to z-normalize
                im_batch[i,:,:,0] = copy.deepcopy(im[:,:,0])#self.zNormalized( copy.deepcopy(im[:,:,0]) )
                im_batch[i,:,:,1] = copy.deepcopy(im[:,:,1])#self.zNormalized( copy.deepcopy(im[:,:,1]) )
                im_batch[i,:,:,2] = copy.deepcopy(im[:,:,2])#self.zNormalized( copy.deepcopy(im[:,:,2]) )
                label_batch[i,0] =  copy.deepcopy( y[0] )
                label_batch[i,1] =  copy.deepcopy( y[1] )
                label_batch[i,2] =  copy.deepcopy( y[2] )
                label_batch[i,3] =  copy.deepcopy( y[3] )

        else:
            f_im = 'im_batch.pickle'
            f_lab = 'label_batch.pickle'
            print 'Loading : ', f_im, f_lab
            with open( f_im, 'rb' ) as handle:
                im_batch = pickle.load(handle )


            with open( f_lab, 'rb' ) as handle:
                label_batch = pickle.load(handle )
            print 'Done.@!'

            # im_batch = copy.deepcopy( self.X_im_batch )
            # # label_batch = copy.deepcopy( self.X_label_batch )
            #
            # r0 = np.random.randint( 0, im_batch.shape[0], batchsize )
            # r1 = np.random.randint( 0, im_batch.shape[0], batchsize )


        # Note:
        # What is being done here is a bit of a hack. The thing is
        # in the mainloop() ie. in train_tf_decop.py doesn't allow any
        # if statements. So, I have instead saved a few example-renders on a
        # pickle-file. If the queue is not sufficiently filled i just return
        # from the saved file.

        return im_batch, label_batch

    ###### TENSORFLOW HELPERS ######
    def define_l2_loss( self, infer_op, label_x, label_y, label_z, label_yaw ):
        """ defines the l2-loss """
        loss_x = tf.reduce_mean( tf.square( tf.sub( infer_op[0], label_x ) ), name='loss_x' )
        loss_y = tf.reduce_mean( tf.square( tf.sub( infer_op[1], label_y ) ), name='loss_y' )
        loss_z = tf.reduce_mean( tf.square( tf.sub( infer_op[2], label_z ) ), name='loss_z' )
        loss_yaw = tf.reduce_mean( tf.square( tf.sub( infer_op[3], label_yaw ) ), name='loss_yaw' )

        xpy = tf.add( tf.mul( loss_x, 1.0 ), tf.mul( loss_y, 1.0 ), name='x__p__y')
        zpyaw = tf.add(tf.mul( loss_z, 1.0 ), tf.mul( loss_yaw, 0.5 ), name='z__p__yaw' )
        loss_total = tf.sqrt( tf.add(xpy,zpyaw,name='full_sum'), name='aggregated_loss'  )
        return loss_total


    def _print_trainable_variables(self):
        """ Print all trainable variables. List is obtained using tf.trainable_variables() """
        var_list = tf.trainable_variables()
        print '--Trainable Variables--', 'length= ', len(var_list)
        total_n_nums = []
        for vr in var_list:
            shape = vr.get_shape().as_list()
            n_nums = np.prod(shape)
            total_n_nums.append( n_nums )
            print self.tcolor.OKGREEN, vr.name, shape, n_nums, self.tcolor.ENDC

        print self.tcolor.OKGREEN, 'Total Trainable Params (floats): ', sum( total_n_nums )
        print 'Not counting the pop_mean and pop_varn as these were set to be non trainable', self.tcolor.ENDC


    ###### END OF TF HELPERS ######



    def __init__(self):
        ShowBase.__init__(self)
        self.taskMgr.add( self.renderNlearnTask, "renderNlearnTask" ) #changing camera poses
        self.taskMgr.add( self.putAxesTask, "putAxesTask" ) #draw co-ordinate axis
        self.taskMgr.add( self.putTrainingBox, "putTrainingBox" )


        # Set up training area. This is used in monte_carlo_sample() and putTrainingBox()
        self.mc_X_max = 300
        self.mc_X_min = -300

        self.mc_Y_max = 350
        self.mc_Y_min = -500

        self.mc_Z_max = 120
        self.mc_Z_min = 35

        self.mc_yaw_max = 60
        self.mc_yaw_min = -60

        self.mc_roll_max = 5
        self.mc_roll_min = -5

        self.mc_pitch_max = 5
        self.mc_pitch_min = -5

        # # Note params
        # self.PARAM_TENSORBOARD_PREFIX = TENSORBOARD_PREFIX
        # self.PARAM_MODEL_SAVE_PREFIX = MODEL_SAVE_PREFIX
        # self.PARAM_MODEL_RESTORE = MODEL_RESTORE
        #
        # self.PARAM_WRITE_SUMMARY_EVERY = WRITE_SUMMARY_EVERY
        # self.PARAM_WRITE_TF_MODEL_EVERY = WRITE_TF_MODEL_EVERY


        # Misc Setup
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.setFrameRateMeter(True)

        self.tcolor = TerminalColors.bcolors()




        #
        # Set up Mesh (including load, position, orient, scale)
        self.setupMesh()
        self.positionMesh()


        # Custom Render
        #   Important Note: self.render displays the low_res and self.scene0 is the images to retrive
        self.scene0 = NodePath("scene0")
        # cytX = copy.deepcopy( cyt )
        self.low_res.reparentTo(self.render)

        self.cyt.reparentTo(self.scene0)
        self.cyt2.reparentTo(self.scene0)





        #
        # Make Buffering Window
        bufferProp = FrameBufferProperties().getDefault()
        props = WindowProperties()
        props.setSize(1280, 960)
        win2 = self.graphicsEngine.makeOutput( pipe=self.pipe, name='wine1',
        sort=-1, fb_prop=bufferProp , win_prop=props,
        flags=GraphicsPipe.BFRequireWindow)
        #flags=GraphicsPipe.BFRefuseWindow)
        # self.window = win2#self.win #dr.getWindow()
        self.win2 = win2
        # self.win2.setupCopyTexture()



        # Adopted from : https://www.panda3d.org/forums/viewtopic.php?t=3880
        #
        # Set Multiple Cameras
        self.cameraList = []
        for i in range(4*4):
            print 'Create camera#', i
            self.cameraList.append( self.customCamera( str(i) ) )


        # Disable default camera
        # dr = self.camNode.getDisplayRegion(0)
        # dr.setActive(0)




        #
        # Set Display Regions (4x4)
        dr_list = self.customDisplayRegion(4,4)


        #
        # Setup each camera
        for i in  range(len(dr_list)):
            dr_list[i].setCamera( self.cameraList[i] )


        #
        # Set buffered Queues (to hold rendered images and their positions)
        # each queue element will be an RGB image of size 240x320x3
        self.q_imStack = Queue.Queue()
        self.q_labelStack = Queue.Queue()

        print 'Panda3d Renderer Initialization Complete'


        # print 'Loaded : im_batch.pickle, label_batch.pickle'
        # with open( 'im_batch.pickle', 'rb' ) as handle:
        #         self.X_im_batch = pickle.load(handle )
        #
        #
        # with open( 'label_batch.pickle', 'rb' ) as handle:
        #         self.X_label_batch = pickle.load(handle )


        #
        # Set up Caffe (possibly in future TensorFLow)
        # caffe.set_device(0)
        # caffe.set_mode_gpu()
        # self.solver = None
        # self.solver = caffe.SGDSolver(solver_proto)
        # self.caffeIter = 0
        # self.caffeTrainingLossX = np.zeros(300000)
        # self.caffeTrainingLossY = np.zeros(300000)
        # self.caffeTrainingLossZ = np.zeros(300000)
        # self.caffeTrainingLossYaw = np.zeros(300000)


        #
        #Set up TensorFlow through puf (PlutoFlow)
        # #multigpu - have all the trainables on CPU
        # puf_obj = puf.PlutoFlow(trainable_on_device='/cpu:0')
        #
        # # #multigpu - SGD Optimizer on cpu
        # with tf.device( '/cpu:0' ):
        #     self.tf_learning_rate = tf.placeholder( 'float', shape=[], name='learning_rate' )
        #     self.tensorflow_optimizer = tf.train.AdamOptimizer( self.tf_learning_rate )
        #
        #
        #
        # # Define Deep Residual Nets
        # # #multigpu - have the infer computation on each gpu (with different batches)
        # self.tf_tower_cost = []
        # self.tf_tower_infer = []
        # tower_grad = []
        # self.tf_tower_ph_x = []
        # self.tf_tower_ph_label_x = []
        # self.tf_tower_ph_label_y = []
        # self.tf_tower_ph_label_z = []
        # self.tf_tower_ph_label_yaw = []
        #
        # for gpu_id in [0,1]:
        #     with tf.device( '/gpu:'+str(gpu_id) ):
        #         with tf.name_scope( 'tower_'+str(gpu_id) ):
        #             # have placeholder `x`, label_x, label_y, label_z, label_yaw
        #             tf_x = tf.placeholder( 'float', [None,240,320,3], name='x' )
        #             tf_label_x = tf.placeholder( 'float',   [None,1], name='label_x')
        #             tf_label_y = tf.placeholder( 'float',   [None,1], name='label_y')
        #             tf_label_z = tf.placeholder( 'float',   [None,1], name='label_z')
        #             tf_label_yaw = tf.placeholder( 'float', [None,1], name='label_yaw')
        #
        #             # infer op
        #             tf_infer_op = puf_obj.resnet50_inference(tf_x, is_training=True)  # Define these inference ops on all the GPUs
        #
        #             # Cost
        #             with tf.variable_scope( 'loss'):
        #                 gpu_cost = self.define_l2_loss( tf_infer_op, tf_label_x, tf_label_y, tf_label_z, tf_label_yaw )
        #
        #             # self._print_trainable_variables()
        #
        #
        #             # Gradient computation op
        #             # following grad variable contain a list of 2 elements each
        #             # ie. ( (grad_v0_gpu0,var0_gpu0),(grad_v1_gpu0,var1_gpu0) ....(grad_vN_gpu0,varN_gpu0) )
        #             tf_grad_compute = self.tensorflow_optimizer.compute_gradients( gpu_cost )
        #
        #
        #             # Make list of tower_cost, gradient, and placeholders
        #             self.tf_tower_cost.append( gpu_cost )
        #             self.tf_tower_infer.append( tf_infer_op )
        #             tower_grad.append( tf_grad_compute )
        #
        #             self.tf_tower_ph_x.append(tf_x)
        #             self.tf_tower_ph_label_x.append(tf_label_x)
        #             self.tf_tower_ph_label_y.append(tf_label_y)
        #             self.tf_tower_ph_label_z.append(tf_label_z)
        #             self.tf_tower_ph_label_yaw.append(tf_label_yaw)
        #
        #
        # self._print_trainable_variables()
        #
        #
        #
        # # Average Gradients (gradient_gpu0 + gradient_gpu1 + ...)
        # with tf.device( '/gpu:0'):
        #     n_gpus = len( tower_grad )
        #     n_trainable_variables = len(tower_grad[0] )
        #     tf_avg_gradient = []
        #
        #     for i in range( n_trainable_variables ): #loop over trainable variables
        #         t_var = tower_grad[0][i][1]
        #
        #         t0_grad = tower_grad[0][i][0]
        #         t1_grad = tower_grad[1][i][0]
        #
        #         # ti_grad = [] #get Gradients from each gpus
        #         # for gpu_ in range( n_gpus ):
        #         #     ti_grad.append( tower_grad[gpu_][i][0] )
        #         #
        #         # grad_total = tf.add_n( ti_grad, name='gradient_adder' )
        #
        #         grad_total = tf.add( t0_grad, t1_grad )
        #
        #         frac = 1.0 / float(n_gpus)
        #         t_avg_grad = tf.mul( grad_total ,  frac, name='gradi_scaling' )
        #
        #         tf_avg_gradient.append( (t_avg_grad, t_var) )
        #
        #
        #
        # with tf.device( '/cpu:0' ):
        #     # Have the averaged gradients from all GPUS here as arg for apply_grad()
        #     self.tensorflow_apply_grad = self.tensorflow_optimizer.apply_gradients( tf_avg_gradient )
        #
        #
        #
        #
        #
        #
        #
        #
        # # Model Saver
        # self.tensorflow_saver = tf.train.Saver()
        #
        # # Fire up the TensorFlow-Session
        # # self.tensorflow_session = tf.Session( config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True) )
        # self.tensorflow_session = tf.Session( config=tf.ConfigProto(allow_soft_placement=True) )
        #
        #
        # # Initi from scratch or load from model
        # if self.PARAM_MODEL_RESTORE == None:
        #     #Start from scratch
        #     print self.tcolor.OKGREEN,'global_variables_initializer() : xavier', self.tcolor.ENDC
        #     self.tensorflow_session.run( tf.global_variables_initializer() )
        # else:
        #     #Restore model
        #     restore_file_name = self.PARAM_MODEL_RESTORE #'tf.models/model-5100'
        #     print self.tcolor.OKGREEN,'restore model', restore_file_name, self.tcolor.ENDC
        #     self.tensorflow_saver.restore( self.tensorflow_session, restore_file_name )
        #
        #
        # tf.train.start_queue_runners(sess=self.tensorflow_session)
        #
        #
        # # Holding variables
        # self.tensorflow_iteration = 0
        #
        #
        # # TensorBoard Writer
        # # self.tensorflow_summary_writer = tf.train.SummaryWriter( self.PARAM_TENSORBOARD_PREFIX, graph=tf.get_default_graph() )
        # # self.tensorflow_summary_op = tf.merge_all_summaries()
        # self.tensorflow_summary_writer = tf.summary.FileWriter( self.PARAM_TENSORBOARD_PREFIX, graph=tf.get_default_graph() )
        #
        #
        #
        #
        #
        #
        # # code.interact(local=locals())
        # print 'Done __init__'
        # code.interact(local=locals())
