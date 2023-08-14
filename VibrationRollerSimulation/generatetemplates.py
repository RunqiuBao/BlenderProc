import blenderproc as bproc
from blenderproc.api import camera

import sys
from scipy.spatial.transform import Rotation as R
import numpy
import os
import cv2
import json
from icosphere import icosphere

from commonutils.datatypeutils import MyJsonEncoder
sys.path.append(os.path.dirname(__file__))
from common import GetRotationMatrixFromTwoVectors


class TemplateRenderer(object):
    def __init__(self, modelPath, objectLightDistance=10, kk=None, imageSize=(720, 1280)):
        bproc.init()
        bproc.loader.load_blend(modelPath)
        if kk == None:
            kk = numpy.array([
                [1.77777818e+03 / 2, 0.00000000e+00, 6.39500000e+02],
                [0.00000000e+00, 1.77777818e+03 / 2, 3.59500000e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]    
            ])
        bproc.camera.set_intrinsics_from_K_matrix(K = kk, image_width = imageSize[1], image_height = imageSize[0])
        bproc.renderer.set_world_background([0, 0, 0])  # always black bkg
        lightList = []
        lightLocations = [
            [objectLightDistance / 2, 0, objectLightDistance],
            [-objectLightDistance / 2, 0, objectLightDistance],
            [0, objectLightDistance / 2, objectLightDistance],
            [0, -objectLightDistance / 2, objectLightDistance],
            [objectLightDistance / 2, 0, -objectLightDistance],
            [-objectLightDistance / 2, 0, -objectLightDistance],
            [0, objectLightDistance / 2, -objectLightDistance],
            [0, -objectLightDistance / 2, -objectLightDistance]
        ]
        lightStrength = 3000
        for lightLocation in lightLocations: 
            light = bproc.types.Light()
            light.set_location(lightLocation)
            light.set_energy(lightStrength)
            lightList.append(light)

    def GetSubdividedIcosahedronVertices(self, level=8, xyzRanges=None):
        '''
        Generate uniformly sampled vertices on a sphere, by subdividing an icosahedron.
        Args:
            level: level of subdivision. How many splits in an edge of icosahedron.
            xyzRanges: dict of {'xMax', 'xMin', 'yMax', 'yMin', 'zMax', 'zMin'}, to filter the vertices.

        Returns:
            filteredVertices
        '''
        if xyzRanges.get('primemeridianonly', False):
            vertices = []
            for indexLevel in range(level + 1):
                geodesicTheta = numpy.pi * indexLevel / level / 2
                vertices.append(numpy.array([numpy.cos(geodesicTheta), 0, numpy.sin(geodesicTheta)]))
            filteredVertices = numpy.array(vertices)
            filteredVertices = [vertex for vertex in filteredVertices if (vertex[0] <= xyzRanges['xMax'] and vertex[0] >= xyzRanges['xMin'] and vertex[1] <= xyzRanges['yMax'] and vertex[1] >= xyzRanges['yMin'] and vertex[2] <= xyzRanges['zMax'] and vertex[2] >= xyzRanges['zMin'])]
            return filteredVertices
        vertices, faces = icosphere(level)
        if xyzRanges is not None:
            filteredVertices = [vertex for vertex in vertices if (vertex[0] <= xyzRanges['xMax'] and vertex[0] >= xyzRanges['xMin'] and vertex[1] <= xyzRanges['yMax'] and vertex[1] >= xyzRanges['yMin'] and vertex[2] <= xyzRanges['zMax'] and vertex[2] >= xyzRanges['zMin'])]
        else:
            filteredVertices = vertices
        return filteredVertices

    def GetVerticesOnSingleLongitude(self, geocentricAngleStep=10, xyzRanges=None):
        '''
        Generate uniformly sampled vertices on a longitude of a sphere.
        Args:
            geocentricAngleStep: step between two vetices.
            xyzRanges: dict of {'xMax', 'xMin', 'yMax', 'yMin', 'zMax', 'zMin'}, to filter the vertices.

        Returns:
            filteredVertices
        '''
        pass

    def GenerateTemplates(self,
        outputPath,
        object2CameraDistance=15,
        isFixRz=False,
        xAxisWithinAngleOfEquatorTangentLine=0,
        camViewpointsSamplingLevel=8,
        yawUpsampleFactor=5,
        camXyzRangesInUnitSphere=None
    ):
        '''
        Generate templates from viewpoints on a unit sphere.
        Args:
            outputPath:
            camXyzRangesInUnitSphere: pos restriction of cams on the unit sphere.
            objectCameraDistance:
            kk:
            imageSize:
        '''
        if isFixRz:
            pass
        else:
            camViewpointsOnUnitSphere = self.GetSubdividedIcosahedronVertices(level=camViewpointsSamplingLevel, xyzRanges=camXyzRangesInUnitSphere)
        print("Totally %d camera viewpoints.", len(camViewpointsOnUnitSphere))
        templateInfos = []
        for indexViewpoint, camViewpoint in enumerate(camViewpointsOnUnitSphere):
            rotationMatrix = GetRotationMatrixFromTwoVectors([0, 0, -1], -camViewpoint)
            translation = object2CameraDistance * camViewpoint
            camInObjectTransformation = bproc.math.build_transformation_mat(translation, [0, 0, 0])
            for indexYaw in range(yawUpsampleFactor):
                camRotationBase = numpy.array([-numpy.pi / 2 + 2 * numpy.pi * indexYaw / yawUpsampleFactor, 0, 0])
                rr = R.from_euler('zyx', camRotationBase)
                camRotationBaseMatrix = rr.as_matrix()
                camInObjectTransformation[:3, :3] = numpy.matmul(rotationMatrix, camRotationBaseMatrix)
                if xAxisWithinAngleOfEquatorTangentLine != 0:
                    xyDirVec = numpy.array([camInObjectTransformation[0, 3], camInObjectTransformation[1, 3]])
                    xyDirVec /= numpy.linalg.norm(xyDirVec)
                    rotate90 = numpy.array([
                        [numpy.cos(numpy.deg2rad(90)), -numpy.sin(numpy.deg2rad(90))],
                        [numpy.sin(numpy.deg2rad(90)), numpy.cos(numpy.deg2rad(90))]
                    ])
                    unitVectorEquatorTangentLine = numpy.concatenate((numpy.matmul(rotate90, xyDirVec), numpy.array([0])), axis=0)
                    if not (numpy.abs(numpy.dot(camInObjectTransformation[:3, 1], unitVectorEquatorTangentLine)) <= numpy.abs(numpy.cos(numpy.deg2rad(90 + 0.5))) and camInObjectTransformation[2, 1] > 0):
                        continue  # pose is tilted, not parallel to quator tangent line
                bproc.camera.add_camera_pose(camInObjectTransformation)
                camInObjectTransformation[:3, 3] /= 10  # Note: cad model in mesh file is somehow 10 times larger than the one in scene.
                templateInfos.append({
                    'camInObjectTransformation': camInObjectTransformation
                })
                break
        data = bproc.renderer.render()
        os.makedirs(outputPath, exist_ok=True)
        margin = 10
        for indexImage, image in enumerate(data['colors']):
            maskImage = numpy.where(image[:, :, 0] >= 10, 255, 0).astype('uint8')  # >=10: rendering noise
            x, y, w, h = cv2.boundingRect(maskImage)
            cv2.imwrite(os.path.join(outputPath, '{}.png').format(str(indexImage).zfill(6)), cv2.cvtColor(image[(y - margin):(y + h + margin), (x - margin):(x + w + margin)], cv2.COLOR_RGB2BGR))
            templateInfos[indexImage]['templId'] = indexImage

        try:
            with open(os.path.join(outputPath, 'templateInfos.json'), 'w') as outfile:
                json.dump(templateInfos, outfile, cls=MyJsonEncoder)
        except Exception as e:
            from IPython import embed; print('here!'); embed()


if __name__ == "__main__":
    myRenderer = TemplateRenderer(modelPath="/home/runqiu/colorCone.blend")
    myRenderer.GenerateTemplates(
        outputPath = './output/templates/',
        xAxisWithinAngleOfEquatorTangentLine=1,
        camViewpointsSamplingLevel=16,
        yawUpsampleFactor=370,
        camXyzRangesInUnitSphere={
            'xMax': 1,
            'xMin': 0,
            'yMax': 1,
            'yMin': -1,
            'zMax': 0.99,
            'zMin': 0,
            'primemeridianonly': True
        }
    )
