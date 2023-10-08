import blenderproc as bproc
from blenderproc.api import camera
"""
Given the current position and orientaion from localization, render stereo camera images.
"""
import os
currentDir = os.path.dirname(__file__)
import sys
sys.path.append(currentDir)  # blenderproc python need this to locate the rest of modules

from common import StereoCamera
from scipy.spatial.transform import Rotation as R

class StereoCameraRenderer(object):
    '''
    Setup the scene in blender, render stereo image every time given camera pose input.
    '''
    _stereoCamera = None
    _blenderCameraHandler = None
    
    def __init__(self, stereoCameraName, imageWidth, imageHeight, kk, baseline, blenderCamera):
        '''
        Args:
            blenderCamera: camera handler from blender for rendering.
        '''
        self._stereoCamera = StereoCamera(stereoCameraName, imageWidth, imageHeight, kk, baseline)
        self._blenderCameraHandler = blenderCamera
        # self._blenderCameraHandler.set_resolution(image_width = imageWidth, image_height = imageHeight)
        self._blenderCameraHandler.set_intrinsics_from_K_matrix(K = kk, image_width = imageWidth, image_height = imageHeight)

    def RenderOnePose(self, leftCameraInWorldTransform, blenderRenderer):
        '''
        Args:
            leftCameraInWorldTransform: as name.
            blenderRenderer: render handler from blender.

        returns:
            leftCamImage.
            rightCamImage.
        '''
        self._blenderCameraHandler.add_camera_pose(leftCameraInWorldTransform)
        self._blenderCameraHandler.add_camera_pose(self._stereoCamera.GetRightCameraInWorldTransform(leftCameraInWorldTransform))
        observations = blenderRenderer.render()
        return observations['colors'][0], observations['colors'][1]


if __name__ == "__main__":
    # test code
    from pathplanning import VibrationRollerPathPlanner
    import cv2
    from scipy.spatial.transform import Rotation as R
    import numpy

    myPathPlanner = VibrationRollerPathPlanner()
    # cameraPoseList = [numpy.eye(4)]  # profile image

    # render test sequence
    bproc.init()

    # Create a simple object:
    scenePath = '/home/runqiu/site2_forevent_3.blend'
    bkgPath = '/mnt/data/blenderMaterials/hdri/resting_place_2_4k.exr'
    obj = bproc.loader.load_blend(scenePath)
    # bproc.object.delete_multiple(obj[-4:])  # delete those baloons
    bproc.world.set_world_background_hdr_img(bkgPath)

    isOnlyGeneratePose = False
    if isOnlyGeneratePose:
        with open("/home/runqiu/tmptmp/test-dataset/ts.txt", "r") as file:
            tsList = file.readlines()
        ts = []
        for line in tsList:
            if line[-1] == '\n':
                ts.append(float(line[:-1]))
            else:
                ts.append(float(line))
        cameraPoseList = myPathPlanner.InitializeRollerPath(
            1, 0,
            camTranslation=None,
            camRotation=None,
            laneWidth=2.0,
            moveStep=0.05,
            ts=ts,
            timeAnchorsIndicies=numpy.array([
                [0, 154],
                [158, 629],
                [631, 787]
            ])
        )
    else:
        cameraPoseList = myPathPlanner.InitializeRollerPath(
            1, 0,
            moveStep=0.002,
            vibrationMagnitude=0.01,
            numStepsHalfVibrationCycle=8
        )

    # Set the camera to be in front of the object
    cameraPoseBase = numpy.array([
        [-6.3081],
        [8.69168],
        [1.15959],
    ])
    camTranslationBase = numpy.array([-6.3081, 8.69168, 1.15959])
    camRotationBase = [0, 0, 0]
    camTranslationLocal = numpy.array([0, 0, 0])
    camRotationLocal = [numpy.pi / 2, 0, -numpy.pi / 2]
    workStartInWorldTransform = bproc.math.build_transformation_mat(camTranslationBase, camRotationBase)
    camViewTransform = bproc.math.build_transformation_mat(camTranslationLocal, camRotationLocal)
    camInWorldTransformList = []
    for cameraInWorkStartTransform in cameraPoseList:
        camInWorldTransform = numpy.matmul(workStartInWorldTransform, cameraInWorkStartTransform)
        camInWorldTransform = numpy.matmul(camInWorldTransform, camViewTransform)
        camInWorldTransformList.append(camInWorldTransform)
    # bproc.camera.set_resolution(image_width=1280, image_height=720)
    kk = numpy.array([
        [1.77777818e+03 / 2, 0.00000000e+00, 6.39500000e+02],
        [0.00000000e+00, 1.77777818e+03 / 2, 3.59500000e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]    
    ])
    bproc.camera.set_intrinsics_from_K_matrix(K = kk, image_width = 1280, image_height = 720)
    # initialize stereo camera
    myStereoCam = StereoCameraRenderer('myStereoCam', int(1280), int(720), kk, 0.8, bproc.camera)

    # Note: auto adjusting KK

    # Render the data
    os.makedirs('/home/runqiu/tmptmp/vslam-4/leftcam', exist_ok=True)
    os.makedirs('/home/runqiu/tmptmp/vslam-4/rightcam', exist_ok=True)
    outputPosePath = '/home/runqiu/tmptmp/vslam-3/gtpose.txt'
    for indexCamPose, camInWorldTransform in enumerate(camInWorldTransformList):
        print('indexCamPose: {}, new frame pos: {}'.format(indexCamPose, camInWorldTransform[:3, 3]))
        if not isOnlyGeneratePose:
            leftImage, rightImage = myStereoCam.RenderOnePose(camInWorldTransform, bproc.renderer)
            cv2.imwrite('/home/runqiu/tmptmp/vslam-4/leftcam/{}.png'.format(str(indexCamPose).zfill(6)), cv2.cvtColor(leftImage, cv2.COLOR_RGB2BGR))
            cv2.imwrite('/home/runqiu/tmptmp/vslam-4/rightcam/{}.png'.format(str(indexCamPose).zfill(6)), cv2.cvtColor(rightImage, cv2.COLOR_RGB2BGR))
        rRotation = R.from_matrix(camInWorldTransform[:3, :3])
        qRotation = rRotation.as_quat()
        onePose = str(indexCamPose) + ' ' + ' '.join(map(str, camInWorldTransform[:3, 3])) + ' ' + ' '.join(map(str, qRotation))
        if indexCamPose == 0:
            with open(outputPosePath, 'w') as f:
                f.write(onePose)
        else:
            with open(outputPosePath, 'a') as f:
                f.write('\n' + onePose)
        bproc.utility.reset_keyframes()

    from IPython import embed; print('here!'); embed()
