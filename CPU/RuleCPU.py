import random
import numpy as np


class Rule:
    def __init__(self, cpuBase):
        self.cpu = cpuBase
        self.nocca = cpuBase.nocca

    def getIput(self):
        prevPoint = None
        nextPoint = None

        # 動かせるコマ
        prevPointCandidate = self.nocca.canMovePiecesPoints()
        maxValue = -1000000
        maxPrevList = []
        maxNextList = []
        breakFlag = False
        for prevP in prevPointCandidate:
            nextPointCandidate = self.nocca.canMovePointsFrom(prevP)
            for nextP in nextPointCandidate:
                # ゴールに入れるならそこを選んで終わり
                if np.all(nextP == self.nocca.MyGoalPoint) or np.all(nextP == self.nocca.OppGoalPoint):
                    maxPrevList = [prevP]
                    maxNextList = [nextP]
                    breakFlag = True
                    break

                tmpValue = self.stateEvaluatedValue(
                    self.nocca.move(prevP, nextP, False))
                if tmpValue > maxValue:
                    maxValue = tmpValue
                    maxPrevList = [prevP]
                    maxNextList = [nextP]
                elif tmpValue == maxValue:
                    maxPrevList.append(prevP)
                    maxNextList.append(nextP)

            if breakFlag:
                break

        randIndex = random.choice(range(len(maxPrevList)))
        prevPoint = maxPrevList[randIndex]
        nextPoint = maxNextList[randIndex]

        return prevPoint, nextPoint

    def stateEvaluatedValue(self, state):
        weightList = []
        valueLIst = []

        # 自己コマのxが相手のゴールに近い(xが小さい)
        weightList.append(1)
        valueLIst.append(self.sumOfX(state))

        # 相手の駒の上にのっている-自分の駒にのっている
        weightList.append(3)
        valueLIst.append(self.numOfOnPieces(state))

        # -相手の駒の隣にいる
        weightList.append(4)
        valueLIst.append(-self.numOfAdjacentPieces(state))

        sumValue = 0
        for i in range(len(weightList)):
            sumValue += valueLIst[i] * weightList[i]
        return sumValue

    def sumOfX(self, state):
        ans = 0
        for x in range(self.nocca.XRANGE):
            for y in range(self.nocca.YRANGE):
                for z in range(self.nocca.ZRANGE):
                    if state[x][y][z] == self.cpu.player:
                        ans += (x + 1 if self.cpu.player == 1 else 6 - x)
        return ans

    def numOfOnPieces(self, state):
        ans = 0
        for x in range(self.nocca.XRANGE):
            for z in range(self.nocca.ZRANGE):
                topState = self.nocca.topState(np.array([x, z]))[0]
                if(topState == self.cpu.player):
                    ans += x + 1 if self.cpu.player == 1 else 6 - x
                    if x == (0 if self.cpu.player == -1 else 5):
                        ans += 6
                elif(topState == -self.cpu.player):
                    ans -= x + 1 if self.cpu.player == 1 else 6 - x
                    if x == (5 if self.cpu.player == -1 else 0):
                        ans -= 6
        return ans

    def numOfAdjacentPieces(self, state):
        ans = 0
        pIteration = [
            [0, 1],
            [0, -1],
            [1, 0],
            [1, 1],
            [1, -1],
            [-1, 0],
            [-1, 1],
            [-1, -1],
        ]
        for x in range(self.nocca.XRANGE):
            for z in range(self.nocca.ZRANGE):
                for pi in pIteration:
                    tmpP = np.array([x, z])
                    # 範囲内
                    if 0 <= (tmpP + pi)[0] and (tmpP + pi)[0] < self.nocca.XRANGE and 0 <= (tmpP + pi)[1] and (tmpP + pi)[1] < self.nocca.ZRANGE:
                        topstate = self.nocca.topState(tmpP + pi)[0]
                        if(topstate == -self.cpu.player):
                            ans += 1
        return ans
