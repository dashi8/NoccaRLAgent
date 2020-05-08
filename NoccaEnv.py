import numpy as np


class NoccaEnv:
    XRANGE = 6
    ZRANGE = 5
    YRANGE = 3
    MyGoalPoint = np.array([-2, -2])
    OppGoalPoint = np.array([-1, -1])

    def __init__(self, firstTurn):
        self.FIRST_TURN = firstTurn  # FIRST_TURN 1手目が自分(1)or相手(-1)
        self.initState()

    def initState(self):
        self.isMyTurn = self.FIRST_TURN
        self.isGameOver = False  # gameoverになったらTrue
        self.winner = 0  # isGameOverがFalseのときは0, Trueのときは勝者

        self.state = np.array([[[0 for z in range(self.ZRANGE)]
                                for y in range(self.YRANGE)] for x in range(self.XRANGE)])
        for z in range(self.ZRANGE):
            self.state[0][0][z] = 1
        for z in range(self.ZRANGE):
            self.state[self.XRANGE - 1][0][z] = -1

    def move(self, prevPoint, nextPoint, canMove):
        prevPoint = np.array(prevPoint)
        nextPoint = np.array(nextPoint)

        # 動かせるポイントが渡されること前提
        # 動かせないときとゲーム終了時はNoneを返す
        # それ以外のときは移動後のstateを返す

        prevTopState = self.topState(prevPoint)
        nextTopState = self.topState(nextPoint)

        # ゲームオーバ済み
        if self.isGameOver:
            return None
        # 動かそうとするコマとisMyTurnが一致しない
        if(prevTopState[0] != self.isMyTurn):
            return None
        # 移動先が埋まってる
        if(nextTopState[1] == self.YRANGE - 1):
            return None
        # 移動先がゴール
        if(np.all(nextPoint == self.MyGoalPoint)):
            self.winner = 1
            self.isGameOver = True
            return None
        elif(np.all(nextPoint == self.OppGoalPoint)):
            self.winner = -1
            self.isGameOver = True
            return None

        # 移動処理
        copiedState = np.copy(self.state)
        copiedState[nextPoint[0]][nextTopState[1] +
                                  1][nextPoint[1]] = prevTopState[0]
        copiedState[prevPoint[0]][prevTopState[1]][prevPoint[1]] = 0
        # 次の人が動かせない（ゲーム終了）
        if self.checkAllPieceCannotMove(copiedState):
            return None
        # canMoveのときだけ移動後のstateをコピー
        # ターン交代
        if canMove:
            self.isMyTurn *= -1
            self.state = copiedState
        return copiedState

    def checkAllPieceCannotMove(self, checkState):
        print("checkAllPieceCannotMove")
        print(self.isMyTurn)
        cannotMove = True
        for x in range(self.XRANGE):
            for z in range(self.ZRANGE):
                tempPoint = np.array([x, z])
                if self.topState(tempPoint)[0] == self.isMyTurn:
                    if len(self.canMovePointsFrom(tempPoint, checkState)) > 0:
                        cannotMove = False
                        break
            if not cannotMove:
                break

        if cannotMove:
            print("winner:{}:cannotMove".format(self.isMyTurn))
            self.isGameOver = True
            self.winner = self.isMyTurn

        return cannotMove

    def canMovePointsFrom(self, fromPoint, cs=None):
        checkState = None
        if cs is None:
            checkState = self.state
        else:
            checkState = cs

        fromPoint = np.array(fromPoint)
        canMovePoints = []

        # ゴール
        if np.all(fromPoint == self.MyGoalPoint) or np.all(fromPoint == self.OppGoalPoint):
            return canMovePoints

        # 手番と駒が一致
        if self.topState(fromPoint, checkState)[0] == self.isMyTurn:
            # 周り8箇所を見る
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
            for pi in pIteration:
                # 範囲内
                if 0 <= (fromPoint + pi)[0] and (fromPoint + pi)[0] < self.XRANGE and 0 <= (fromPoint + pi)[1] and (fromPoint + pi)[1] < self.ZRANGE:
                    # 飽和してない
                    if self.topState(fromPoint + pi, checkState)[1] != self.YRANGE - 1:
                        canMovePoints.append(fromPoint + pi)
            # ゴールを追加
            if self.topState(fromPoint, checkState)[0] == 1 and fromPoint[0] == self.XRANGE - 1:
                canMovePoints.append(self.MyGoalPoint)
            elif self.topState(fromPoint, checkState)[0] == -1 and fromPoint[0] == 0:
                canMovePoints.append(self.OppGoalPoint)

        return canMovePoints

    def canMovePiecesPoints(self):
        canMovePoints = []
        for x in range(self.XRANGE):
            for z in range(self.ZRANGE):
                checkingPoint = np.array([x, z])
                if self.topState(checkingPoint)[0] == self.isMyTurn:
                    canMovePoints.append(checkingPoint)
        return canMovePoints

    def topState(self, point, sc=None):
        checkState = None
        if sc is None:
            checkState = self.state
        else:
            checkState = sc
        topstate = 0
        step = -1
        # ゴールの時
        if np.all(point == self.MyGoalPoint) or np.all(point == self.OppGoalPoint):
            return [0, 0]
        # 手番とコマが一致
        for y in reversed(range(self.YRANGE)):
            if checkState[point[0]][y][point[1]] != 0:
                topstate = checkState[point[0]][y][point[1]]
                step = y
                break
        return [topstate, step]

    def getState(self):
        return np.copy(self.state)

    def getWinner(self):
        return self.winner

    # 現ターンの駒の場所すべて
    def getAllPiecesPoints(self):
        pointList = []
        for x in range(self.XRANGE):
            for y in range(self.YRANGE):
                for z in range(self.ZRANGE):
                    if self.state[x][y][z] == self.isMyTurn:
                        pointList.append(np.array([x, z]))
        return pointList

    def render(self):
        print("  ", end="")
        for z in range(self.ZRANGE):
            print("{}    ".format(z), end="")
        print()
        for x in range(self.XRANGE):
            print("{} ".format(x), end="")
            for z in range(self.ZRANGE):
                print("[", end="")
                for y in range(self.YRANGE):
                    if self.state[x][y][z] == 0:
                        print(" ", end="")
                    elif self.state[x][y][z] == 1:
                        print("o", end="")
                    elif self.state[x][y][z] == -1:
                        print("x", end="")
                print("]", end="")
            print("\n")


if __name__ == "__main__":
    env = NoccaEnv(True)

    env.render()
