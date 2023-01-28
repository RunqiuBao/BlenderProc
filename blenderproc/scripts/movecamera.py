import blenderproc as bproc
from blenderproc.api import camera

"""
The quickstart example:

1. A monkey object is created plus a light, which illuminates the monkey.
2. A light is created, placed and gets a proper energy level set
3. The camera is placed in the scene to look at the monkey
4. A color image is rendered
5. The rendered image is saved in an .hdf5 file container

"""
import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as R

bproc.init()

# Create a simple object:
scenePath = '/home/runqiu/site.blend'
bkgPath = '/media/runqiu/data/blenderMaterials/hdri/moonless_golf_4k.jpg'
obj = bproc.loader.load_blend(scenePath)
# bproc.object.delete_multiple(obj[-4:])  # delete those baloons
bproc.world.set_world_background_hdr_img(bkgPath)

# import bpy
# bpy.context.scene.render.engine = 'BLENDER_EEVEE'  # 'CYCLES'
# bpy.context.scene.eevee.use_bloom = True


# Create a point light next to it
# light = bproc.types.Light()
# light.set_location([2, -2, 0])
# light.set_energy(300)

# Set the camera to be in front of the object
cameraPoseBase = np.array([
    [-0.07231751829385757, 0.24019721150398254, -0.9680265784263611, -9.255331993103027],
    [-0.995455801486969, -0.07766836136579514, 0.055094730108976364, 7.211221218109131],
    [-0.06195143982768059, 0.9676119685173035, 0.24472248554229736, 2.187476634979248],
    [0.0, 0.0, 0.0, 1.0]
])
camTranslationBase = cameraPoseBase[:3, 3]
rMatrix = R.from_matrix(cameraPoseBase[:3, :3])
camRotationBaseRaw = rMatrix.as_euler('zyx', degrees=False)
camRotationBase = [np.pi / 2, 0, - np.pi / 2]
for ii in range(6):
    camTranslation = camTranslationBase
    camTranslation[0] += ii * 0.05
    cam_pose = bproc.math.build_transformation_mat(camTranslation, camRotationBase)
    bproc.camera.add_camera_pose(cam_pose)
# bproc.camera.set_resolution(image_width=1280, image_height=720)
kk = np.array([
    [1.77777818e+03 / 2, 0.00000000e+00, 6.39500000e+02],
    [0.00000000e+00, 1.77777818e+03 / 2, 3.59500000e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]    
])
bproc.camera.set_intrinsics_from_K_matrix(K = kk, image_width = 1280, image_height = 720)

# Note: auto adjusting KK

# Render the scene
data = bproc.renderer.render()
os.makedirs('output/seq2/', exist_ok=True)
for indexImage, image in enumerate(data['colors']):
    cv2.imwrite('output/seq2/{}.png'.format(str(indexImage).zfill(6)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
 
# # Write the rendering into a hdf5 file
# bproc.writer.write_hdf5("output/", data)
