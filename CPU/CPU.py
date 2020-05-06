from .RandomCPU import Random
from .RuleCPU import Rule


class CPU:
    def __init__(self, player, nocca, policy_type):
        self.player = player
        self.nocca = nocca
        self.policy = self.getPolicy(policy_type)

    def getPolicy(self, policy_type):
        if policy_type == "Random":
            return Random(self)
        elif policy_type == "Rule":
            return Rule(self)
        else:
            return None

    # [prevPoint, nextPoint]を返す
    def getIput(self):
        return self.policy.getIput()
