""" The training-render file. (class TrainRenderer)
        Update: This class is render only class. It does not take the mainloop()
        control. Basically need to call function step().

        Defines a rendering class. Defines a spinTask (panda3d) which basicalyl
        renders 16-cameras at a time and sets them into a CPU-queue. This queue
        is emtied by calls to step(). May build more purpose-build steps()s

        update: There are 3 Panda renderer in this file, viz. TrainRenderer,
        TestRenderer, NetVLADRenderer. For comments on what each does check out
        those classes. For general usage of the renderers see `test_render.py`

"""

# Panda3d
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.stdpy.threading2 import Thread

# Usual Math and Image Processing
import numpy as np
import cv2
from scipy import interpolate

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
import PathMaker


class TrainRenderer(ShowBase):
    renderIndx=0


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



    ## Gives a random 6-dof pose. Need to set params manually here.
    ## X,Y,Z,  Yaw(abt Z-axis), Pitch(abt X-axis), Roll(abt Y-axis)
    ## @param No : no inputs
    def monte_carlo_sample(self):

        # mc_X_min etc are set in constructor
        X = np.random.uniform(self.mc_X_min,self.mc_X_max)
        Y = np.random.uniform(self.mc_Y_min,self.mc_Y_max)
        Z = np.random.uniform(self.mc_Z_min,self.mc_Z_max)

        yaw = np.random.uniform( self.mc_yaw_min, self.mc_yaw_max)
        roll = 0#np.random.uniform( self.mc_roll_min, self.mc_roll_max)
        pitch = 0#np.random.uniform( self.mc_pitch_min, self.mc_pitch_max)

        return X,Y,Z, yaw,roll,pitch

    ## Annotation-helpers for self.render
    def putBoxes(self,X,Y,Z,r=1.,g=0.,b=0., scale=1.0):
        cube_x = CubeMaker.CubeMaker().generate()
        cube_x.setColor(r,g,b)
        cube_x.setScale(scale)
        cube_x.reparentTo(self.render)
        cube_x.setPos(X,Y,Z)

    ## Set a cube in 3d env
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


    ## Task. This task draw the XYZ axis
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


    ## Render-n-Learn task
    ##
    ## set pose in each camera <br/>
    ## make note of the poses just set as this will take effect next <br/>
    ## Retrive Rendered Data <br/>
    ## Cut rendered data into individual image. Note rendered data will be 4X4 grid of images <br/>
    ## Put imX into the queue <br/>
    def renderNlearnTask(self, task):
        if task.time < 2: #do not do anything for 1st 2 sec
            return task.cont


        # print randX, randY, randZ

        #
        ## set pose in each camera
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



        ## make note of the poses just set as this will take effect next
        if TrainRenderer.renderIndx == 0:
            TrainRenderer.renderIndx = TrainRenderer.renderIndx + 1
            self.prevPoses = poses
            return task.cont



        #
        ## Retrive Rendered Data
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
        ## Cut rendered data into individual image. Note rendered data will be 4X4 grid of images
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

                    ## Put imX into the queue
                    # do not queue up if queue size begin to exceed 150


                    self.q_imStack.put( imX )
                    self.q_labelStack.put( texPoses[c,:] )


                    # fname = '__'+str(poses[c,0]) + '_' + str(poses[c,1]) + '_' + str(poses[c,2]) + '_' + str(poses[c,3]) + '_'
                    # cv2.imwrite( str(TrainRenderer.renderIndx)+'__'+str(i)+str(j)+fname+'.png', imX.astype('uint8') )

                    c = c + 1
        else:
            if self.queue_warning:
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


    ## Execute 1-step.
    ##
    ## This function is to be called from outside to render once. This is a wrapper for app.taskMgr.step()
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
        if self.q_imStack.qsize() < 16*5:
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
            return None, None
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
            r0 = np.random.randint( 0, im_batch.shape[0], batchsize )
            # r1 = np.random.randint( 0, im_batch.shape[0], batchsize )
            im_batch = im_batch[r0]
            label_batch = label_batch[r0]

        # Note:
        # What is being done here is a bit of a hack. The thing is
        # in the mainloop() ie. in train_tf_decop.py doesn't allow any
        # if statements. So, I have instead saved a few example-renders on a
        # pickle-file. If the queue is not sufficiently filled i just return
        # from the saved file.

        return im_batch, label_batch




    def __init__(self, queue_warning=True):
        ShowBase.__init__(self)
        self.taskMgr.add( self.renderNlearnTask, "renderNlearnTask" ) #changing camera poses
        self.taskMgr.add( self.putAxesTask, "putAxesTask" ) #draw co-ordinate axis
        self.taskMgr.add( self.putTrainingBox, "putTrainingBox" )

        self.queue_warning = queue_warning #suppress the warning of queue full if this var is True


        # Set up training area. This is used in monte_carlo_sample() and putTrainingBox()
        self.mc_X_max = 300
        self.mc_X_min = -300

        self.mc_Y_max = 360
        self.mc_Y_min = -360

        self.mc_Z_max = 120
        self.mc_Z_min = 45

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



        print self.tcolor.OKGREEN, '\n##########\n'+'Panda3d Renderer Initialization Complete'+'\n##########\n', self.tcolor.ENDC





class TestRenderer(ShowBase):
    renderIndx=0

    ## Basic Mesh & Camera Setup
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

        self.low_res = self.loader.loadModel( 'model_l/l0/level_0_0_0.obj' )
        print self.tcolor.OKGREEN, 'Done Loading Vertices', self.tcolor.ENDC

        print 'Attempt Loading Textures'
        self.loadAllTextures( self.cyt, 'model_l/l6/')
        self.loadAllTextures( self.cyt2, 'model_l/l6/')
        self.loadAllTextures( self.low_res, 'model_l/l0/')
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



    def monte_carlo_sample(self):
        """ Gives a random 6-dof pose. Need to set params manually here.
                X,Y,Z,  Yaw(abt Z-axis), Pitch(abt X-axis), Roll(abt Y-axis) """
        X = np.random.uniform(-50,50)
        Y = np.random.uniform(-100,100)
        Z = np.random.uniform(50,100)

        yaw = np.random.uniform(-60,60)
        roll = np.random.uniform(-5,5)
        pitch = np.random.uniform(-5,5)

        return X,Y,Z, yaw,roll,pitch

    ## Annotation-helpers for self.render
    def putBoxes(self,X,Y,Z,r=1.,g=0.,b=0., scale=1.0):
        cube_x = CubeMaker.CubeMaker().generate()
        cube_x.setColor(r,g,b)
        cube_x.setScale(scale)
        cube_x.reparentTo(self.render)
        cube_x.setPos(X,Y,Z)
    def putAxesTask(self,task):


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


    ## Render-n-Learn task.
    ##      Sets the camera position (1-cam only) with spline or your choice (see init).
    ###     Renders the image at that point and queue the image and its position (pose)
    def renderNtestTask(self, task):
        if task.frame < 50: #do not do anything for 50 ticks, as spline's 1st node is at t=50
            return task.cont


        # print randX, randY, randZ
        t = task.frame
        if t > self.spl_u.max():
            print 'End of Spline, End task'
            # fName = 'trace__' + self.pathGen.__name__ + '.npz'
            # np.savez( fName, loss=self.loss_ary, gt=self.gt_ary, pred=self.pred_ary )
            # print 'PathData File Written : ', fName
            # print 'Visualize result : `python tools/analyse_path_trace_subplot.py', fName, '`'
            return None


        #
        # set pose in each camera
        # Note: The texture is grided images in a col-major format
        # TODO : since it is going to be only 1 camera eliminate this loop to simply code
        poses = np.zeros( (len(self.cameraList), 4), dtype='float32' )
        for i in range(len(self.cameraList)): #here usually # of cams will be 1 (for TestRenderer)
            indx = TestRenderer.renderIndx
            pt = interpolate.splev( t, self.spl_tck)
            #randX,randY, randZ, randYaw, randPitch, randRoll = self.monte_carlo_sample()

            randX = pt[0]
            randY = pt[1]
            randZ = pt[2]
            randYaw = pt[3]
            randPitch = 0
            randRoll = 0


            self.cameraList[i].setPos(randX,randY,randZ)
            self.cameraList[i].setHpr(randYaw,-90+randPitch,0+randRoll)

            poses[i,0] = randX
            poses[i,1] = randY
            poses[i,2] = randZ
            poses[i,3] = randYaw




        # make note of the poses just set as this will take effect next
        if TestRenderer.renderIndx == 0:
            TestRenderer.renderIndx = TestRenderer.renderIndx + 1
            # self.putBoxes(0,0,0, scale=100)
            self.prevPoses = poses
            return task.cont




        #
        # Retrive Rendered Data
        tex = self.win2.getScreenshot()
        # A = np.array(tex.getRamImageAs("RGB")).reshape(960,1280,3) #@#
        A = np.array(tex.getRamImageAs("RGB")).reshape(240,320,3)
        # A = np.zeros((960,1280,3))
        # A_bgr =  cv2.cvtColor(A.astype('uint8'),cv2.COLOR_RGB2BGR)
        # cv2.imwrite( str(TestRenderer.renderIndx)+'.png', A_bgr.astype('uint8') )
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
        # TODO : Eliminate this 2-loop as we know there is only 1 display region
        #if self.q_imStack.qsize() < 150: #no limit on queue size
            # for j in range(1): #j is for cols-indx
                # for i in range(1): #i is for rows-indx
        i=0
        j=0
        #print i*nRows, j*nCols, (i+1)*nRows, (j+1)*nCols
        im = A[i*nRows:(i+1)*nRows,j*nCols:(j+1)*nCols,:]
        #imX = im.astype('float32')/255. - .5 # TODO: have a mean image
        #imX = (im.astype('float32') - 128.0) /128.
        imX = im.astype('float32')  #- self.meanImage


        # Put imX into the queue
        # do not queue up if queue size begin to exceed 150


        self.q_imStack.put( imX )
        self.q_labelStack.put( texPoses[c,:] )
        self.putBoxes( texPoses[c,0], texPoses[c,1], texPoses[c,2] )
        # print 'putBoxes', texPoses[c,0], texPoses[c,1], texPoses[c,2]

        # fname = '__'+str(poses[c,0]) + '_' + str(poses[c,1]) + '_' + str(poses[c,2]) + '_' + str(poses[c,3]) + '_'
        # cv2.imwrite( str(TestRenderer.renderIndx)+'__'+str(i)+str(j)+fname+'.png', imX.astype('uint8') )

        c = c + 1



        #
        # Prep for Next Iteration
        TestRenderer.renderIndx = TestRenderer.renderIndx + 1
        self.prevPoses = poses

        # if( TestRenderer.renderIndx > 5 ):
            # return None

        return task.cont


    def step(self):
        self.taskMgr.step()

        # print 'Queues Status (imStack=%d,labelStack=%d)' %(self.q_imStack.qsize(), self.q_labelStack.qsize())

        # Dequeue 1 elements
        if self.q_imStack.qsize() > 2: # Do not dequeue if the queue size is less than 2
            im = copy.deepcopy( self.q_imStack.get() ) #240x320x3 RGB
            y = copy.deepcopy( self.q_labelStack.get() )

            return im, y
        else:
            return None, None


    def __init__(self, pathGen=None ):
        ShowBase.__init__(self)
        self.taskMgr.add( self.renderNtestTask, "renderNtestTask" ) #changing camera poses
        self.taskMgr.add( self.putAxesTask, "putAxesTask" ) #draw co-ordinate axis



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
        # props.setSize(1280, 960)
        props.setSize(320, 240) #@#
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
        # for i in range(4*4):
        for i in range(1*1): #@#
            print 'Create camera#', i
            self.cameraList.append( self.customCamera( str(i) ) )


        # Disable default camera
        # dr = self.camNode.getDisplayRegion(0)
        # dr.setActive(0)




        #
        # Set Display Regions (4x4)
        dr_list = self.customDisplayRegion(1,1)


        #
        # Setup each camera
        for i in  range(len(dr_list)):
            dr_list[i].setCamera( self.cameraList[i] )


        #
        # Set buffered Queues (to hold rendered images and their positions)
        # each queue element will be an RGB image of size 240x320x3
        self.q_imStack = Queue.Queue()
        self.q_labelStack = Queue.Queue()



        #
        # Setting up Splines
        # Note: Start interpolation at 50,
        if pathGen is None:
            # self.pathGen = PathMaker.PathMaker().path_flat_h
            # self.pathGen = PathMaker.PathMaker().path_smallM
            # self.pathGen = PathMaker.PathMaker().path_yaw_only
            # self.pathGen = PathMaker.PathMaker().path_bigM
            # self.pathGen = PathMaker.PathMaker().path_flat_spiral
            # self.pathGen = PathMaker.PathMaker().path_helix
            # self.pathGen = PathMaker.PathMaker().path_like_real
            # self.pathGen = PathMaker.PathMaker().path_like_real2
            self.pathGen = PathMaker.PathMaker().path_large_loop
        else:
            self.pathGen = pathGen

        t,X = self.pathGen()

        self.spl_tck, self.spl_u = interpolate.splprep(X.T, u=t.T, s=0.0, per=1)

        print 'Test Renderer Init Done'
        print self.tcolor.OKGREEN, 'Test Renderer Init Done', self.tcolor.ENDC



# Setup NetVLAD Renderer - This renderer is custom made for NetVLAD training
# It renders 16 images at a time. (q, (P1,P2,..P5), (N1,N2,...,N10)).
# ie. 1st image is im, next 5 are near this im (potential positives).
# Last 10 are far from im (definite negatives)
class NetVLADRenderer(ShowBase):
    renderIndx=0


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


    def mc_default( self, cam0X, cam0Y, cam0Z ):
        return 0,0,80,0,0,0

    def mc_far( self, cam0X, cam0Y, cam0Z ):
        rf = np.random.uniform
        nZ = rf(self.mc_Z_min,self.mc_Z_max)
        fov = 1.3962 #80 degrees
        sigma = cam0Z * np.tan(fov/2.)/3


        nX = rf(self.mc_X_min, cam0X - 2*sigma) if rf(-1,1) > 0 else rf(cam0X + 2*sigma, self.mc_X_max )
        nY = rf(self.mc_Y_min, cam0Y - 2*sigma) if rf(-1,1) > 0 else rf(cam0Y + 2*sigma, self.mc_Y_max )

        yaw = rf(self.mc_yaw_min, self.mc_yaw_max)
        return nX, nY, nZ, yaw, 0 , 0

    # Return a random sample near (cam0X,cam0Y,cam0Z)
    def mc_near( self, cam0X, cam0Y, cam0Z ):
        rf = np.random.uniform
        nZ = rf(self.mc_Z_min,self.mc_Z_max)
        fov = 1.3962 #80 degrees
        sigma = cam0Z * np.tan(fov/2.)/3

        yaw = rf(self.mc_yaw_min, self.mc_yaw_max)
        return rf(cam0X-sigma,cam0X+sigma),  rf(cam0Y-sigma,cam0Y+sigma),nZ,yaw,0.,0.

    ## Gives a random 6-dof pose. Need to set params manually here.
    ## X,Y,Z,  Yaw(abt Z-axis), Pitch(abt X-axis), Roll(abt Y-axis)
    ## @param No
    def monte_carlo_sample(self):

        # mc_X_min etc are set in constructor
        X = np.random.uniform(self.mc_X_min,self.mc_X_max)
        Y = np.random.uniform(self.mc_Y_min,self.mc_Y_max)
        Z = np.random.uniform(self.mc_Z_min,self.mc_Z_max)

        yaw = np.random.uniform( self.mc_yaw_min, self.mc_yaw_max)
        roll = 0#np.random.uniform( self.mc_roll_min, self.mc_roll_max)
        pitch = 0#np.random.uniform( self.mc_pitch_min, self.mc_pitch_max)

        return X,Y,Z, yaw,roll,pitch

    ## Annotation-helpers for self.render
    def putBoxes(self,X,Y,Z,r=1.,g=0.,b=0., scale=1.0):
        cube_x = CubeMaker.CubeMaker().generate()
        cube_x.setColor(r,g,b)
        cube_x.setScale(scale)
        cube_x.reparentTo(self.render)
        cube_x.setPos(X,Y,Z)

    ## Set a cube in 3d env
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


    ## Task. This task draw the XYZ axis
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


    ## Render-n-Learn task
    ##
    ## set pose in each camera <br/>
    ## make note of the poses just set as this will take effect next <br/>
    ## Retrive Rendered Data <br/>
    ## Cut rendered data into individual image. Note rendered data will be 4X4 grid of images <br/>
    ## Put imX into the queue <br/>
    def renderNlearnTask(self, task):
        if task.time < 2: #do not do anything for 1st 2 sec
            return task.cont


        # print randX, randY, randZ

        #
        ## set pose in each camera
        # Note: The texture is grided images in a col-major format
        poses = np.zeros( (len(self.cameraList), 4), dtype='float32' )
        _randX= _randY= _randZ= _randYaw= _randPitch= _randRoll = 0
        for i in range(len(self.cameraList)):

            if i==0:
                _randX,_randY, _randZ, _randYaw, _randPitch, _randRoll = self.monte_carlo_sample()
                (randX,randY, randZ, randYaw, randPitch, randRoll) = _randX, _randY, _randZ, _randYaw, _randPitch, _randRoll
            elif i>=1 and i<6:
                randX,randY, randZ, randYaw, randPitch, randRoll = self.mc_near(_randX, _randY, _randZ )
            else:
                randX,randY, randZ, randYaw, randPitch, randRoll = self.mc_far(_randX, _randY, _randZ)



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



        ## make note of the poses just set as this will take effect next
        if NetVLADRenderer.renderIndx == 0:
            NetVLADRenderer.renderIndx = NetVLADRenderer.renderIndx + 1
            self.prevPoses = poses
            return task.cont



        #
        ## Retrive Rendered Data
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
        ## Cut rendered data into individual image. Note rendered data will be 4X4 grid of images
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

                    # print 'Noise Added to renderedIm'
                    # imX =  imX + 10.*np.random.randn( imX.shape[0], imX.shape[1], imX.shape[2] )


                    ## Put imX into the queue
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
        NetVLADRenderer.renderIndx = NetVLADRenderer.renderIndx + 1
        self.prevPoses = poses



        return task.cont


    ## Execute 1-step.
    ##
    ## This function is to be called from outside to render once. This is a wrapper for app.taskMgr.step()
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
            return None, None
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
            r0 = np.random.randint( 0, im_batch.shape[0], batchsize )
            # r1 = np.random.randint( 0, im_batch.shape[0], batchsize )
            im_batch = im_batch[r0]
            label_batch = label_batch[r0]

        # Note:
        # What is being done here is a bit of a hack. The thing is
        # in the mainloop() ie. in train_tf_decop.py doesn't allow any
        # if statements. So, I have instead saved a few example-renders on a
        # pickle-file. If the queue is not sufficiently filled i just return
        # from the saved file.

        return im_batch, label_batch




    def __init__(self):
        ShowBase.__init__(self)
        self.taskMgr.add( self.renderNlearnTask, "renderNlearnTask" ) #changing camera poses
        self.taskMgr.add( self.putAxesTask, "putAxesTask" ) #draw co-ordinate axis
        self.taskMgr.add( self.putTrainingBox, "putTrainingBox" )


        # Set up training area. This is used in monte_carlo_sample() and putTrainingBox()
        self.mc_X_max = 300
        self.mc_X_min = -300

        self.mc_Y_max = 360
        self.mc_Y_min = -360

        self.mc_Z_max = 120
        self.mc_Z_min = 45

        self.mc_yaw_max = 85
        self.mc_yaw_min = -85

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



        print self.tcolor.OKGREEN, '\n##########\n'+'Panda3d Renderer Initialization Complete'+'\n##########\n', self.tcolor.ENDC
