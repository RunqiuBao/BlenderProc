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
import shutil
import pickle
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

    isOnlyGeneratePose = True
    if isOnlyGeneratePose:
        with open("/home/runqiu/tmptmp/vslam-2/ts-blur.txt", "r") as file:
            tsList = file.readlines()
        ts = []
        for line in tsList:
            if line[-1] == '\n':
                ts.append(float(line[:-1]))
            else:
                ts.append(float(line))
        cameraPoseList = myPathPlanner.InitializeRollerPath(
            4, 4, 0,
            laneWidth=2.0,
            moveStep=0.005,
            ts=ts,
            timeAnchorsIndicies=numpy.array([
                # seq2
                # [0, 154],
                # [158, 629],
                # [631, 787]
                # seq0
                # [0, 1545],
                # [1546, 6245],
                # [6246, 7772]
                # seq2, vslam
                [0, 401],
                [402, 1601],
                [1602, 2003]
            ])
        )
    else:
        cameraPoseList = myPathPlanner.InitializeRollerPath(
            4, 4, 0,
            # 4, 4, 0,
            laneWidth=2.0,
            moveStep=0.01
            # moveStep=0.005,
            # vibrationMagnitude=0.1,
            # numStepsHalfVibrationCycle=4
        )

    # Set the camera to be in front of the object
    # seq 1
    # cameraPoseBase = numpy.array([
    #     [-0.07231751829385757, 0.24019721150398254, -0.9680265784263611, -9.255331993103027],
    #     [-0.995455801486969, -0.07766836136579514, 0.055094730108976364, 7.211221218109131],
    #     [-0.06195143982768059, 0.9676119685173035, 0.24472248554229736, 2.187476634979248],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    # seq 2
    cameraPoseBase = numpy.array([
        [-9.255331993103027],
        [7.211221218109131],
        [2.187476634979248],
    ])
    # profile image
    # cameraPoseBase = numpy.array([
        # [-0.07231751829385757, 0.24019721150398254, -0.9680265784263611, -18.255331993103027],  # Note: - is back move
        # [-0.995455801486969, -0.07766836136579514, 0.055094730108976364, 3.211221218109131],   # Note: - is right move
        # [-0.06195143982768059, 0.9676119685173035, 0.24472248554229736, 3.187476634979248],
        # [0.0, 0.0, 0.0, 1.0]
    # ])
    camTranslationBase = numpy.array([-9.255331993103027, 7.211221218109131, 2.187476634979248])
    camRotationBase = [0, 0, 0]
    camTranslationLocal = numpy.array([0, 0, 0])
    camRotationLocal = [numpy.pi / 2, 0, -numpy.pi / 2]
    workStartInWorldTransform = bproc.math.build_transformation_mat(camTranslationBase, camRotationBase)
    camViewTransform = bproc.math.build_transformation_mat(camTranslationLocal, camRotationLocal)
    camInWorldTransformList = []
    for cameraInWorkStartTransform in cameraPoseList:
        # camTranslationInWorld = cameraInWorkStartTransform[:3, 3] + workStartInWorldTransform[:3, 3]
        # camRotationInWorld = numpy.matmul(cameraInWorkStartTransform[:3, :3], workStartInWorldTransform[:3, :3])
        # camInWorldTransform = numpy.eye(4)
        # camInWorldTransform[:3, :3] = camRotationInWorld
        # camInWorldTransform[:3, 3] = camTranslationInWorld
        camInWorldTransform = numpy.matmul(workStartInWorldTransform, cameraInWorkStartTransform)
        camInWorldTransform = numpy.matmul(camInWorldTransform, camViewTransform)
        camInWorldTransformList.append(camInWorldTransform)
    # bproc.camera.set_resolution(image_width=1280, image_height=720)
    kk = numpy.array([
        [1.77777818e+03 / 2, 0.00000000e+00, 6.39500000e+02],
        [0.00000000e+00, 1.77777818e+03 / 2, 3.59500000e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]    
    ])

    if isOnlyGeneratePose:
        # save gt poses
        outputPosePath = '/home/runqiu/tmptmp/vslam-2/gtpose_timealigned.txt'
        for indexCamPose, camInWorldTransform in enumerate(camInWorldTransformList):
            print('indexCamPose: {}, new frame pos: {}'.format(indexCamPose, camInWorldTransform[:3, 3]))
            rRotation = R.from_matrix(camInWorldTransform[:3, :3])
            qRotation = rRotation.as_quat()
            onePose = str(indexCamPose) + ' ' + ' '.join(map(str, camInWorldTransform[:3, 3])) + ' ' + ' '.join(map(str, qRotation))
            if indexCamPose == 0:
                with open(outputPosePath, 'w') as f:
                    f.write(onePose)
            else:
                with open(outputPosePath, 'a') as f:
                    f.write('\n' + onePose)
    else:
        # save to n groups for parallel rendering
        numProcess = 12
        numMinPosesInGroup = int(len(camInWorldTransformList) / numProcess)
        listPoseGroup = []
        listPoseGroupIndices = []
        indexAllPoses = 0
        for indexProcess in range(numProcess):
            posesOneGroup = []
            posesIndicesOneGroup = []
            for indexInGroup in range(numMinPosesInGroup):
                posesOneGroup.append(camInWorldTransformList[indexAllPoses])
                posesIndicesOneGroup.append(indexAllPoses)
                indexAllPoses += 1
            listPoseGroup.append(posesOneGroup)
            listPoseGroupIndices.append(posesIndicesOneGroup)
        listPoseGroup[0].extend(camInWorldTransformList[indexAllPoses:])
        listPoseGroupIndices[0].extend(range(indexAllPoses, len(camInWorldTransformList)))
        if os.path.exists("renderingGroupData"):
            # Remove the folder and its contents
            shutil.rmtree("renderingGroupData")
        os.makedirs("renderingGroupData", exist_ok=False)
        for indexGroup in range(numProcess):
            dictDataGroup = {}
            dictDataGroup['kk'] = kk
            dictDataGroup['poses'] = listPoseGroup[indexGroup]
            dictDataGroup['poseIndices'] = listPoseGroupIndices[indexGroup]
            with open('renderingGroupData/{}.pkl'.format(str(indexGroup).zfill(6)), 'wb') as pickle_file:
                pickle.dump(dictDataGroup, pickle_file)

