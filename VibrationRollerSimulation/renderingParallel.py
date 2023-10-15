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
import numpy
import cv2
import pickle
from scipy.spatial.transform import Rotation as R
import time

from rendering import StereoCameraRenderer

if __name__ == "__main__":
    import argparse
    
    scriptName = os.path.basename(__file__)

    parser = argparse.ArgumentParser(
        description="render one group of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples: \n" +
                '%s --datapath .../000000.pkl/ \n' % scriptName
        )
    )

    parser.add_argument('--datapath', '-i', action='store', type=str, dest='datapath',
                        help='Path to the data in pickle format.')

    args, remaining = parser.parse_known_args()

    with open(args.datapath, 'rb') as pickle_file:
        dictDataGroup = pickle.load(pickle_file)
    kk = dictDataGroup['kk']
    poses = dictDataGroup['poses']
    poseIndices = dictDataGroup['poseIndices']

    # render test sequence
    bproc.init()

    # Create a simple object:
    scenePath = '/home/runqiu/site2_forevent.blend'
    bkgPath = '/mnt/data/blenderMaterials/hdri/moonless_golf_4k.jpg'
    obj = bproc.loader.load_blend(scenePath)
    # bproc.object.delete_multiple([obj[-1]])  # delete those baloons
    # bproc.object.delete_multiple([obj[-4]])  # delete those baloons

    # bproc.world.set_world_background_hdr_img(bkgPath)
    bproc.renderer.set_world_background([0, 0, 0])

    bproc.camera.set_intrinsics_from_K_matrix(K = kk, image_width = 1280, image_height = 720)
    # initialize stereo camera
    myStereoCam = StereoCameraRenderer('myStereoCam', int(1280), int(720), kk, 0.8, bproc.camera)

    # Note: auto adjusting KK

    # Render the data
    os.makedirs('/home/runqiu/tmptmp/vslam-0/leftcam/', exist_ok=True)
    os.makedirs('/home/runqiu/tmptmp/vslam-0/rightcam/', exist_ok=True)
    for indexToIndices, camInWorldTransform in enumerate(poses):
        print('indexCamPose: {}, new frame pos: {}'.format(poseIndices[indexToIndices], camInWorldTransform[:3, 3]))
        # rotateDownTransform = numpy.array([
        #     [1, 0, 0, 0],
        #     [0, 0.9848, 0.1736, 0],
        #     [0, -0.1736, 0.9848, 0],
        #     [0, 0, 0, 1]
        # ])  # Note: rotate down by 15 deg
        # camInWorldTransform = numpy.matmul(camInWorldTransform, rotateDownTransform)
        leftImage, rightImage = myStereoCam.RenderOnePose(camInWorldTransform, bproc.renderer)
        starttime = time.time()
        cv2.imwrite('/home/runqiu/tmptmp/vslam-0/leftcam/{}.png'.format(str(poseIndices[indexToIndices]).zfill(6)), cv2.cvtColor(leftImage, cv2.COLOR_RGB2BGR))
        cv2.imwrite('/home/runqiu/tmptmp/vslam-0/rightcam/{}.png'.format(str(poseIndices[indexToIndices]).zfill(6)), cv2.cvtColor(rightImage, cv2.COLOR_RGB2BGR))
        print("save time cost: {} sec".format(time.time() - starttime))
        bproc.utility.reset_keyframes()

    print("rendering process ({}) finished successfully!".format(args.datapath))
