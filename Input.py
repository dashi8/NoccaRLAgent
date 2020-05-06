import numpy as np


class Input:
    def __init__(self, nocca):
        self.nocca = nocca

    # prevPoint, nextPointを返す
    def getIput(self):
        self.nocca.render()
        print("From:")
        canMovePiecesPoints = self.nocca.canMovePiecesPoints()
        print(canMovePiecesPoints)

        prevPoint = None
        while prevPoint is None:
            tmpInput = input()
            tmpInput = tmpInput.split(" ")
            tmpInput = np.array([int(tmpInput[0]), int(tmpInput[1])])
            for canmove in canMovePiecesPoints:
                if np.all(canmove == tmpInput):
                    prevPoint = tmpInput
                    break

        print("To:")
        canMovePointsFrom = self.nocca.canMovePointsFrom(prevPoint)
        print(canMovePointsFrom)

        nextPoint = None
        while nextPoint is None:
            tmpInput = input()
            tmpInput = tmpInput.split(" ")
            tmpInput = np.array([int(tmpInput[0]), int(tmpInput[-1])])
            for canmove in canMovePointsFrom:
                if np.all(canmove == tmpInput):
                    nextPoint = tmpInput
                    break
        return prevPoint, nextPoint
