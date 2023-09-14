"""
Common base tools and classes
"""
import numpy
import importlib
from commonutils.datatypeutils import MyJsonEncoder, MyJsonDecoder


class Camera(object):
    """
    Camera base class
    """
    _kk = None
    _imageWidth = None
    _imageHeight = None
    _name = None
    _type = None

    def __init__(self, name, type, kk, imageWidth, imageHeight):
        self._name = name
        self._type = type
        self._imageHeight = imageHeight
        self._imageWidth = imageWidth

    @property
    def kk(self):
        return self._kk

    @property
    def imageWidth(self):
        return self._imageWidth

    @property
    def imageWidth(self):
        return self._imageWidth


class StereoCamera(Camera):
    """
    Stereo camera class
    """
    _rightCameraInLeftCameraTransform = None
    # _leftCamera = None
    # _rightCamera = None

    def __init__(self, name, imageWidth, imageHeight, kk, baseline):
        super(StereoCamera, self).__init__(name, 'stereo', kk, imageWidth, imageHeight)
        self._rightCameraInLeftCameraTransform = numpy.eye(4)
        self._rightCameraInLeftCameraTransform[0, 3] = baseline

    def GetRightCameraInWorldTransform(self, leftCameraInWorldTransform):
        return numpy.matmul(leftCameraInWorldTransform, self._rightCameraInLeftCameraTransform)


class CacheManager(object):
    @staticmethod
    def DumpCache(data, cacheFilePath, dumpType='json'):
        if dumpType == 'json':
            dumper = importlib.import_module(dumpType)
            with open(cacheFilePath, 'w') as outfile:
                dumper.dump(data, outfile, cls=MyJsonEncoder)
        else:
            raise NotImplementedError

    @staticmethod
    def LoadCache(cacheFilePath, dumpType='json'):
        loader = importlib.import_module(dumpType)
        with open(cacheFilePath, 'r') as cacheFile:
            data = loader.load(cacheFile, cls=MyJsonDecoder)
        return data


# -------- basic math --------
def GetRotationMatrixFromTwoVectors(v1, v2):
    """
    Compute a matrix R that rotates v1 to align with v2.
    v1 and v2 must be length-3 1d numpy arrays.
    """
    # unit vectors
    u = v1 / numpy.linalg.norm(v1)
    Ru = v2 / numpy.linalg.norm(v2)
    # dimension of the space and identity
    dim = u.size
    I = numpy.identity(dim)
    # the cos angle between the vectors
    c = numpy.dot(u, Ru)
    # a small number
    eps = 1.0e-10
    if numpy.abs(c - 1.0) < eps:
        # same direction
        return I
    elif numpy.abs(c + 1.0) < eps:
        # opposite direction
        return -I
    else:
        # the cross product matrix of a vector to rotate around
        K = numpy.outer(Ru, u) - numpy.outer(u, Ru)
        # Rodrigues' formula
        return I + K + (K @ K) / (1 + c)


if __name__=="__main__":
    print("test script")

