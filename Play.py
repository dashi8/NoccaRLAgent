from NoccaEnv import NoccaEnv
from Input import Input
from CPU.CPU import CPU


def main():
    nocca = NoccaEnv(1)
    myInputGenerator = Input(nocca)
    oppInputGenerator = CPU(player=-1, nocca=nocca, policy_type="Rule")

    while nocca.winner == 0:
        prevPoint = None
        nextPoint = None
        if(nocca.isMyTurn == 1):
            prevPoint, nextPoint = myInputGenerator.getIput()
        elif(nocca.isMyTurn == -1):
            prevPoint, nextPoint = oppInputGenerator.getIput()

        nocca.move(prevPoint, nextPoint, True)

    print("!!!!!!!!!!END!!!!!!!!!!!")
    nocca.render()


if __name__ == "__main__":
    main()
