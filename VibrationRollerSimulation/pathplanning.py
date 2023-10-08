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
from tracemalloc import start
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

    def __init__(self):
        """   
        Args:
            squareTheta: angle between east and square width, 0~90 deg.
            camTranslation: vector3f
            camRotationL: vector3f
        
        Returns:
            poseList: a list of camera poses. The starting pose is identity pose.
        """
        pass

    def InitializeRollerPath(
            self,
            distanceToGo,
            squareTheta,
            moveStep,
            vibrationMagnitude=0.0,
            numStepsHalfVibrationCycle=4,
            ts=None,
            timeAnchorsIndicies=None
        ):
        """
        New mode: if ts provided, generate path of one cycle according to the timestamps.
        timeAnchors including 3 pair, 6 time points in one cycle. 4x2 shape.
        *         *
        ||        |
        |  |      |
        |     |   |
        |       | |
        |        ||
        |         *
        """
        poseList = []
        numStepForward = math.ceil(distanceToGo / moveStep)
        dirForward = numpy.array([numStepForward * moveStep * math.cos(numpy.pi / 2 - math.radians(squareTheta)), -1 * numStepForward * moveStep * math.sin(numpy.pi / 2 - math.radians(squareTheta)), 0])  # x is in SquareHeight direction, y is in minus SquareWidth direction.
        dirForward = dirForward / numpy.linalg.norm(dirForward)
        startPose = numpy.eye(4)
        startPose[:3, :3] = numpy.array([
            [numpy.cos(squareTheta), -numpy.sin(squareTheta), 0],
            [numpy.sin(squareTheta), numpy.cos(squareTheta), 0],
            [0, 0, 1]
        ])
        poseList.append(startPose)
        if vibrationMagnitude > 0:
            sinFuncFactor = numpy.pi / numStepsHalfVibrationCycle
            for i in range(4 * numStepsHalfVibrationCycle + 1):
                newPose = copy.deepcopy(startPose)
                newPose[2, 3] = vibrationMagnitude * numpy.sin(sinFuncFactor * i)
                poseList.append(newPose)
            countStep = 0

        if ts is not None:
            # first lane
            timeSpan = ts[timeAnchorsIndicies[0, 1]] - ts[timeAnchorsIndicies[0, 0]]
            startPose = poseList[-1]
            for indexT in range(timeAnchorsIndicies[0, 0] + 1, timeAnchorsIndicies[0, 1] + 1):
                newPose = copy.deepcopy(startPose)
                newPose[:3, 3] += dirForward * distanceToGo * (ts[indexT] - ts[timeAnchorsIndicies[0, 0]]) / timeSpan
                poseList.append(newPose)
            # add static poses
            if timeAnchorsIndicies[1, 0] > timeAnchorsIndicies[0, 1]:
                staticPose = copy.deepcopy(poseList[-1])
                for time in range(timeAnchorsIndicies[1, 0] - timeAnchorsIndicies[0, 1] - 1):
                    poseList.append(staticPose)
        else:
            for indexStepForward in range(numStepForward):
                newPose = copy.deepcopy(poseList[-1])
                newPose[:3, :3] = poseList[-1][:3, :3]
                newPose[:3, 3] += dirForward * moveStep
                poseList.append(newPose)
                if vibrationMagnitude > 0:
                    newPose[2, 3] = vibrationMagnitude * numpy.sin(sinFuncFactor * countStep)  # Note: z is the height direction.
                    countStep += 1
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
