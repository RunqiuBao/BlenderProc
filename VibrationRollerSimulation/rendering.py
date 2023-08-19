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

    myPathPlanner = VibrationRollerPathPlanner(8, 4, 0, moveStep=0.1)
    cameraPoseList = myPathPlanner.GetPoseList()
    # cameraPoseList = [numpy.eye(4)]  # profile image

    # render test sequence
    bproc.init()

    # Create a simple object:
    scenePath = '/home/runqiu/site2.blend'
    bkgPath = '/mnt/data/blenderMaterials/hdri/moonless_golf_4k.jpg'
    obj = bproc.loader.load_blend(scenePath)
    # bproc.object.delete_multiple(obj[-4:])  # delete those baloons
    bproc.world.set_world_background_hdr_img(bkgPath)

    # Set the camera to be in front of the object
    cameraPoseBase = numpy.array([
        [-0.07231751829385757, 0.24019721150398254, -0.9680265784263611, -9.255331993103027],
        [-0.995455801486969, -0.07766836136579514, 0.055094730108976364, 7.211221218109131],
        [-0.06195143982768059, 0.9676119685173035, 0.24472248554229736, 2.187476634979248],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # profile image
    # cameraPoseBase = numpy.array([
        # [-0.07231751829385757, 0.24019721150398254, -0.9680265784263611, -18.255331993103027],  # Note: - is back move
        # [-0.995455801486969, -0.07766836136579514, 0.055094730108976364, 3.211221218109131],   # Note: - is right move
        # [-0.06195143982768059, 0.9676119685173035, 0.24472248554229736, 3.187476634979248],
        # [0.0, 0.0, 0.0, 1.0]
    # ])
    camTranslationBase = cameraPoseBase[:3, 3]
    rMatrix = R.from_matrix(cameraPoseBase[:3, :3])
    camRotationBaseRaw = rMatrix.as_euler('zyx', degrees=False)
    camRotationBase = [numpy.pi / 2, 0, - numpy.pi / 2]
    workStartInWorldTransform = bproc.math.build_transformation_mat(camTranslationBase, camRotationBase)
    camInWorldTransformList = []
    for cameraInWorkStartTransform in cameraPoseList:
        camTranslationInWorld = cameraInWorkStartTransform[:3, 3] + workStartInWorldTransform[:3, 3]
        camRotationInWorld = numpy.matmul(cameraInWorkStartTransform[:3, :3], workStartInWorldTransform[:3, :3])
        camInWorldTransform = numpy.eye(4)
        camInWorldTransform[:3, :3] = camRotationInWorld
        camInWorldTransform[:3, 3] = camTranslationInWorld
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
    os.makedirs('output/seq/leftCam', exist_ok=True)
    os.makedirs('output/seq/rightCam', exist_ok=True)
    outputPosePath = 'output/seq/pose.txt'
    for indexCamPose, camInWorldTransform in enumerate(camInWorldTransformList):
        print('indexCamPose: {}, new frame pos: {}'.format(indexCamPose, camInWorldTransform[:3, 3]))
        leftImage, rightImage = myStereoCam.RenderOnePose(camInWorldTransform, bproc.renderer)
        cv2.imwrite('output/seq/leftCam/{}.png'.format(str(indexCamPose).zfill(6)), cv2.cvtColor(leftImage, cv2.COLOR_RGB2BGR))
        cv2.imwrite('output/seq/rightCam/{}.png'.format(str(indexCamPose).zfill(6)), cv2.cvtColor(rightImage, cv2.COLOR_RGB2BGR))
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
