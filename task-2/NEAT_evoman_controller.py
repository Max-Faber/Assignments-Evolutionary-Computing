import sys, numpy as np

sys.path.insert(0, 'evoman')
from controller import Controller


class NEATController(Controller):
    def __init__(self, enemy_hint, ff_network):
        self.ffNetwork = ff_network
        self.enemy_hint = enemy_hint
        pass

    def control(self, inputs, controller):
        # Normalises the input using min-max scaling
        inputs = np.append(inputs, controller.enemy_hint)
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        output = controller.ffNetwork.activate(inputs)

        # takes decisions about sprite actions (sigmoid activations so 0.5 boundary)
        left = 1 if output[0] > 0.5 else 0
        right = 1 if output[1] > 0.5 else 0
        jump = 1 if output[2] > 0.5 else 0
        shoot = 1 if output[3] > 0.5 else 0
        release = 1 if output[4] > 0.5 else 0
        return [left, right, jump, shoot, release]
