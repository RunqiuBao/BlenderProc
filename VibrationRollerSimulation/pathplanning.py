"""
Plan the motion of the camera (on a industrial machine). 
Based on the localization result, tell the render where is the camera next step.
|         |         |         |
|  |      |  |      |  |      |
|     |   |     |   |     |   |
|       | |       | |       | |
|        ||        ||        ||
|         |         |         |
|         |         |         |
"""
import numpy
import math
import os
import copy
from scipy.spatial.transform import Rotation
import sympy
from sympy import symbols, Eq, solve

from common import GetRotationMatrixFromTwoVectors, CacheManager


class VibrationRollerPathPlanner(object):
    """
    Given the square area, plan the lane paths.
    The roller starts from the south west corner.
    """
    _poseList = None

    def __init__(self, squareWidth, squareHeight, squareTheta, camTranslation=None, camRotation=None, laneWidth=2.0, moveStep=1.0):
        """   
        Args:
            squareTheta: angle between east and square width, 0~90 deg.
            camTranslation: vector3f
            camRotationL: vector3f
        
        Returns:
            poseList: a list of camera poses. The starting pose is identity pose.
        """
        self._poseList = self.InitializeRollerPath(squareWidth, squareHeight, squareTheta, camTranslation, camRotation, laneWidth, moveStep)

    def InitializeRollerPath(self, squareWidth, squareHeight, squareTheta, camTranslation, camRotation, laneWidth, moveStep):
        numLane = math.ceil(squareWidth / laneWidth)
        poseList = []
        numStepForward = math.ceil(squareHeight / moveStep)
        dirForward = numpy.array([numStepForward * moveStep * math.sin(numpy.pi / 2 - math.radians(squareTheta)), numStepForward * moveStep * math.cos(numpy.pi / 2 - math.radians(squareTheta)), 0])  # x is in SquareHeight direction, y is in minus SquareWidth direction.
        dirForward = dirForward / numpy.linalg.norm(dirForward)
        rotationMatrixHorizontal = Rotation.from_rotvec(math.radians(squareTheta) * numpy.array([0, 0, 1]))
        rotationMatrixHorizontal = rotationMatrixHorizontal.as_matrix()
        numStepsBack = int(squareHeight / moveStep) * 3  # Note: backward need to be slow. Because it contains rotation.
        numStepsBack = numStepsBack if numStepsBack % 2 == 0 else (numStepsBack + 1)
        startPose = numpy.eye(4)
        poseList.append(startPose)
        # generate backward move vectors if cache does not exist.
        backwardVectorCacheFile = os.path.join('/tmp', 'blenderProc-runqiu', 'backwardSmooth.json')
        os.makedirs('/tmp/blenderProc-runqiu/', exist_ok=True)
        if os.path.exists(backwardVectorCacheFile):
            backwardDirsList = CacheManager.LoadCache(backwardVectorCacheFile)['backwardDirsList']
            backwardLocationsList = CacheManager.LoadCache(backwardVectorCacheFile)['backwardLocationsList']
        else:    
            backwardDirsList = []
            backwardLocationsList = []
            firstFunc = [-0.25, 2, -4]
            dFirstFunc = [-0.5, 2]
            secondFunc = [0.25, 0, -2]
            dSecondFunc = [0.5, 0]

            for indexStep in range(numStepsBack + 1):
                x0 = squareHeight - squareHeight / numStepsBack * indexStep
                if indexStep <= (numStepsBack / 2):
                    y0 = firstFunc[0] * x0**2 + firstFunc[1] * (x0) + firstFunc[2]
                    backwardLocationsList.append(numpy.array([x0, y0, 0]))
                    k = dFirstFunc[0] * x0 + dFirstFunc[1]
                    backwardDir = numpy.array([numpy.cos(numpy.arctan(k)), numpy.sin(numpy.arctan(k)), 0])
                    backwardDirsList.append(backwardDir / numpy.linalg.norm(backwardDir))
                else:
                    y0 = secondFunc[0] * x0**2 + secondFunc[1] * (x0) + secondFunc[2]
                    backwardLocationsList.append(numpy.array([x0, y0, 0]))
                    k = dSecondFunc[0] * x0 + dSecondFunc[1]
                    backwardDir = numpy.array([numpy.cos(numpy.arctan(k)), numpy.sin(numpy.arctan(k)), 0])
                    backwardDirsList.append(backwardDir / numpy.linalg.norm(backwardDir))
            CacheManager.DumpCache({'backwardDirsList': backwardDirsList, 'backwardLocationsList': backwardLocationsList}, backwardVectorCacheFile)
        for indexLane in range(numLane):
            for indexStepForward in range(numStepForward):
                newPose = copy.deepcopy(poseList[-1])
                newPose[:3, :3] = poseList[-1][:3, :3]
                newPose[:3, 3] += dirForward * moveStep
                poseList.append(newPose)
            orientationForward = copy.copy(poseList[-1][:3, :3])
            # backward
            backStartPose = copy.deepcopy(poseList[-1])
            for indexBackward in range(numStepsBack + 1):
                newPose = copy.deepcopy(backStartPose)
                newPose[:3, 3] = numpy.matmul(rotationMatrixHorizontal, backwardLocationsList[indexBackward])
                newPose[1:3, 3] += backStartPose[1:3, 3]
                newPose[:3, :3] = numpy.matmul(GetRotationMatrixFromTwoVectors(dirForward, backwardDirsList[indexBackward]), orientationForward)  # need rotation matrix from two vector
                poseList.append(newPose)
            newPose = copy.deepcopy(poseList[-1])
            newPose[:3, :3] = copy.deepcopy(orientationForward)
            poseList.append(newPose)
        return poseList

    def GetPoseList(self):
        return self._poseList

    def SolveIntersection(self, l, m, N, x0, y0):
        '''
        Args:
            l: laneWidth. right root of the virtual vertical ellipse as well.
            m: moveStep
            N: numSteroForward
            x0: current x coordinate
            y0: current y coordinate
        '''
        x, y = symbols('x y')
        eq1 = Eq(x**2 / l**2 + y**2 / (m * N)**2 - 1, 0)
        eq2 = Eq((x - x0)**2 + (y - y0)**2 - m**2, 0)
        roots = solve((eq1, eq2), (x, y))
        correctRootStack = []
        for root in roots:
            rootx = root[0].evalf()
            rooty = root[1].evalf()
            try:
                if isinstance(rootx, sympy.core.numbers.Float) and isinstance(rooty, sympy.core.numbers.Float) and rootx > x0 and rooty < y0:
                    correctRootStack.append((rootx, rooty))
            except Exception as e:
                from IPython import embed; print('here!'); embed()
        if len(correctRootStack) == 1:
            if correctRootStack[0][1] < 0:
                return numpy.array([l, 0]).astype('float')
            else:
                return numpy.array([correctRootStack[0][0], correctRootStack[0][1]]).astype('float')
        elif len(correctRootStack) == 0:
            return numpy.array([l, 0]).astype('float')
        else:
            print("unexpected root happened.")
            from IPython import embed; print('here!'); embed()       


if __name__ == "__main__":
    myPathPlanner = VibrationRollerPathPlanner(40, 20 , 0, None, None, 2, 1)
    aa = myPathPlanner.GetPoseList()
    from IPython import embed; print('here!'); embed()
