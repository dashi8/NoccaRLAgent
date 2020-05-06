import random


class Random:
    def __init__(self, cpuBase):
        self.nocca = cpuBase.nocca

    def getIput(self):
        prevPoint = None
        nextPoint = None

        canMovePiecesPoints = self.nocca.canMovePiecesPoints()
        if len(canMovePiecesPoints) == 0:
            print("gameoverになってるはず")
            self.nocca.render()
        prevPoint = random.choice(canMovePiecesPoints)
        nextPoint = random.choice(
            self.nocca.canMovePointsFrom(prevPoint))

        return prevPoint, nextPoint
