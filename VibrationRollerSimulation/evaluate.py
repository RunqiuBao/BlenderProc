import open3d
import numpy
from scipy.spatial.transform import Rotation as R

def LoadTraj(format, pathToTxt):
    trajPoses = []
    if format == 'tum':
        with open(pathToTxt, 'r') as trajFile:
            trajListStr = trajFile.readlines()
        for line in trajListStr:
            if line[-1] == "\n":
                line = line[:-1]
            sLine = line.split(" ")
            r = R.from_quat([float(sLine[-1]), float(sLine[-4]), float(sLine[-3]), float(sLine[-2])])
            rmatrix = r.as_matrix()
            onePose = numpy.eye(4)
            onePose[:3, :3] = rmatrix


if __name__ == "__main__":
    aaa