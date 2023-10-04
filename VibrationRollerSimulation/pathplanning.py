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
from itertools import count
import numpy
import math
import os
import copy
from scipy.spatial.transform import Rotation
import sympy
from sympy import symbols, Eq, solve
import random

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
        
        Returns:
            poseList: a list of camera poses. The starting pose is identity pose.
        """
        pass

    def InitializeRollerPath(
            self,
            squareWidth,
            squareHeight,
            squareTheta,
            laneWidth,
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

        Args:
            ...
            squareTheta: angle between east and square width, 0~90 deg.
            ...
            vibrationMagnitude: float
            numStepsHalfVibrationCycle: int, half cycle of vibration will last for 3 moveStep.
        """
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
        startPose[:3, :3] = numpy.array([
            [numpy.cos(squareTheta), -numpy.sin(squareTheta), 0],
            [numpy.sin(squareTheta), numpy.cos(squareTheta), 0],
            [0, 0, 1]
        ])
        poseList.append(startPose)

        if vibrationMagnitude > 0:
            sinFuncFactor = numpy.pi / numStepsHalfVibrationCycle
            countStep = 0
        if ts is not None:
            # first lane
            timeSpan = ts[timeAnchorsIndicies[0, 1]] - ts[timeAnchorsIndicies[0, 0]]
            startPose = poseList[-1]
            for indexT in range(timeAnchorsIndicies[0, 0] + 1, timeAnchorsIndicies[0, 1] + 1):
                newPose = copy.deepcopy(startPose)
                newPose[:3, 3] += dirForward * squareHeight * (ts[indexT] - ts[timeAnchorsIndicies[0, 0]]) / timeSpan
                poseList.append(newPose)
            # add static poses
            if timeAnchorsIndicies[1, 0] > timeAnchorsIndicies[0, 1]:
                staticPose = copy.deepcopy(poseList[-1])
                for time in range(timeAnchorsIndicies[1, 0] - timeAnchorsIndicies[0, 1] - 1):
                    poseList.append(staticPose)
            # second (backward)
            firstFunc = [-0.25, 2, -4]
            dFirstFunc = [-0.5, 2]
            secondFunc = [0.25, 0, -2]
            dSecondFunc = [0.5, 0]
            timeSpan = ts[timeAnchorsIndicies[1, 1]] - ts[timeAnchorsIndicies[1, 0]]
            startPose = poseList[-1]
            orientationForward = copy.deepcopy(startPose[:3, :3])
            for indexT in range(timeAnchorsIndicies[1, 0], timeAnchorsIndicies[1, 1] + 1):
                x0 = squareHeight - squareHeight * (ts[indexT] - ts[timeAnchorsIndicies[1, 0]]) / timeSpan
                if ts[indexT] <= (ts[timeAnchorsIndicies[1, 0]] + timeSpan / 2):
                    y0 = firstFunc[0] * x0**2 + firstFunc[1] * (x0) + firstFunc[2]
                    k = dFirstFunc[0] * x0 + dFirstFunc[1]
                else:
                    y0 = secondFunc[0] * x0**2 + secondFunc[1] * (x0) + secondFunc[2]
                    k = dSecondFunc[0] * x0 + dSecondFunc[1]
                backwardLocation = numpy.array([x0, y0, 0])
                backwardDir = numpy.array([numpy.cos(numpy.arctan(k)), numpy.sin(numpy.arctan(k)), 0])
                backwardDir = backwardDir / numpy.linalg.norm(backwardDir)
                newPose = numpy.eye(4)
                newPose[:3, 3] = numpy.matmul(rotationMatrixHorizontal, backwardLocation)
                newPose[1:3, 3] += startPose[1:3, 3]  # Note: if the lane is biased from 0, need to add the starting value of y.
                newPose[:3, :3] = numpy.matmul(GetRotationMatrixFromTwoVectors(numpy.array([1, 0, 0]), backwardDir), orientationForward)
                poseList.append(newPose)
            # add static poses
            if timeAnchorsIndicies[2, 0] > timeAnchorsIndicies[1, 1]:
                staticPose = copy.deepcopy(poseList[-1])
                for time in range(timeAnchorsIndicies[2, 0] - timeAnchorsIndicies[1, 1] - 1):
                    poseList.append(staticPose)
            # third lane
            timeSpan = ts[timeAnchorsIndicies[2, 1]] - ts[timeAnchorsIndicies[2, 0]]
            startPose = poseList[-1]
            for indexT in range(timeAnchorsIndicies[2, 0], timeAnchorsIndicies[2, 1] + 1):
                newPose = copy.deepcopy(startPose)
                newPose[:3, 3] += dirForward * squareHeight * (ts[indexT] - ts[timeAnchorsIndicies[2, 0]]) / timeSpan
                newPose[:3, :3] = orientationForward
                poseList.append(newPose)
        else:
            # generate backward move dirs
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

            randomMagnitude = (random.random() + 1.0) * vibrationMagnitude
            for indexLane in range(numLane):
                for indexStepForward in range(numStepForward):
                    newPose = copy.deepcopy(poseList[-1])
                    newPose[:3, :3] = poseList[-1][:3, :3]
                    newPose[:3, 3] += dirForward * moveStep
                    if vibrationMagnitude > 0:
                        newPose[2, 3] = randomMagnitude * numpy.sin(sinFuncFactor * countStep)  # Note: z is the height direction.
                        countStep += 1
                        if countStep % numStepsHalfVibrationCycle == 0:
                            randomMagnitude = (random.random() + 1.0) * vibrationMagnitude
                    poseList.append(newPose)
                orientationForward = copy.copy(poseList[-1][:3, :3])
                # backward
                backStartPose = copy.deepcopy(poseList[-1])
                for indexBackward in range(numStepsBack + 1):
                    newPose = copy.deepcopy(backStartPose)
                    newPose[:3, 3] = numpy.matmul(rotationMatrixHorizontal, backwardLocationsList[indexBackward])
                    newPose[1, 3] += backStartPose[1, 3]
                    if vibrationMagnitude > 0:
                        newPose[2, 3] = randomMagnitude * numpy.sin(sinFuncFactor * countStep)
                        countStep += 1
                        if countStep % numStepsHalfVibrationCycle == 0:
                            randomMagnitude = (random.random() + 1.0) * vibrationMagnitude
                    newPose[:3, :3] = numpy.matmul(GetRotationMatrixFromTwoVectors(dirForward, backwardDirsList[indexBackward]), orientationForward)  # need rotation matrix from two vector # ? this will break when squareTheta is not 0?
                    poseList.append(newPose)
                newPose = copy.deepcopy(poseList[-1])
                newPose[:3, :3] = copy.deepcopy(orientationForward)
                poseList.append(newPose)  # Note: stop at place for one pose.
        return poseList

# automize backward size:
"""
    # generate backward move dirs
    backwardDirsList = []
    backwardLocationsList = []
    # firstFunc = [-0.25, 2, -4]
    # dFirstFunc = [-0.5, 2]
    # secondFunc = [0.25, 0, -2]
    # dSecondFunc = [0.5, 0]
    firstFunc = [2*squareHeight/3/laneWidth**2, 4*squareHeight/3/laneWidth, squareHeight]
    dFirstFunc = [2*2*squareHeight/3/laneWidth**2, 4*squareHeight/3/laneWidth]
    secondFunc = [-2*squareHeight/3/laneWidth**2, 0, 2*squareHeight/3]
    dSecondFunc = [-4*squareHeight/3/laneWidth**2, 0]

    for indexStep in range(numStepsBack + 1):
        y0Base = 0 - laneWidth / numStepsBack * indexStep
        if indexStep <= (numStepsBack / 2):
            x0 = firstFunc[0] * y0Base**2 + firstFunc[1] * (y0Base) + firstFunc[2]
            backwardLocationsList.append(numpy.array([x0, y0Base, 0]))
            k = dFirstFunc[0] * y0Base + dFirstFunc[1]
            backwardDir = numpy.array([numpy.sin(numpy.arctan(k)), numpy.cos(numpy.arctan(k)), 0])
            backwardDirsList.append(backwardDir / numpy.linalg.norm(backwardDir))
        else:
            x0 = secondFunc[0] * y0Base**2 + secondFunc[1] * (y0Base) + secondFunc[2]
            backwardLocationsList.append(numpy.array([x0, y0Base, 0]))
            k = dSecondFunc[0] * y0Base + dSecondFunc[1]
            backwardDir = numpy.array([numpy.sin(numpy.arctan(k)), numpy.cos(numpy.arctan(k)), 0])
            backwardDirsList.append(backwardDir / numpy.linalg.norm(backwardDir))
"""

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
