from distutils.log import error
import open3d
import os
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
            onePose[:3, 3] = numpy.array([
                float(sLine[1]),
                float(sLine[2]),
                float(sLine[3])
            ])
            trajPoses.append(onePose)
    else:
        raise NotImplementedError

    return trajPoses


def ComputeRMSATE(points, gtpoints, mode):
    if isinstance(points, list):
        points = numpy.concatenate(points, axis=0)
    if isinstance(gtpoints, list):
        gtpoints = numpy.concatenate(gtpoints, axis=0)
    print("====================")
    if mode == '3d':
        pass
    elif mode == '2d':
        points[:, 2] = gtpoints[:, 2]
    errors = numpy.linalg.norm(points - gtpoints, axis=1)
    minError = numpy.min(errors)
    maxError = numpy.max(errors)
    meanError = numpy.mean(errors)
    rmsate = numpy.sqrt(numpy.sum(numpy.power(errors, 2)) / points.shape[0])
    if mode == '3d':
        print("RMS-ATE (m) between gt and tracked traj. in 3d is:\n     ", rmsate)
    elif mode == '2d':
        print("RMS-ATE between gt and tracked traj. in 2d is:\n     ", rmsate)
    print("min translation error (m):  ", minError)
    print("max translation error (m):  ", maxError)
    print("mean translation error (m):  ", meanError)
    print("====================")


# ======================== visualize ============================
def VisualizeTrajCompare(points, gtpoints):
    if isinstance(points, list):
        points = numpy.concatenate(points, axis=0)
    if isinstance(gtpoints, list):
        gtpoints = numpy.concatenate(gtpoints, axis=0)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(numpy.tile(numpy.array([1, 0, 0]), (points.shape[0], 1)))
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(gtpoints)
    pcd2.colors = open3d.utility.Vector3dVector(numpy.tile(numpy.array([0, 0, 1]), (gtpoints.shape[0], 1)))
    open3d.visualization.draw_geometries([pcd, pcd2])


if __name__ == "__main__":
    import argparse
    
    scriptName = os.path.basename(__file__)

    parser = argparse.ArgumentParser(
        description="Evaluate result of SLAM traj.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples: \n" +
                '%s --dataroot .../mydataset/ \n' % scriptName
        )
    )

    parser.add_argument('--dataroot', '-i', action='store', type=str, dest='dataroot',
                        help='Path to the data root.')

    args, remaining = parser.parse_known_args()

    gtPoses = LoadTraj('tum', os.path.join(args.dataroot, 'gtpose_timealigned.txt'))
    trackedPoses = LoadTraj('tum', os.path.join(args.dataroot, 'cameraTrack.txt'))

    gtpoints = []
    for gtPose in gtPoses:
        gtpoints.append(gtPose[:3, 3][numpy.newaxis, :])
    trackpoints = []
    for trackedPose in trackedPoses:
        trackpoints.append(trackedPose[:3, 3][numpy.newaxis, :])
    # VisualizeTrajCompare(trackpoints, gtpoints)
    ComputeRMSATE(trackpoints, gtpoints, mode='3d')
