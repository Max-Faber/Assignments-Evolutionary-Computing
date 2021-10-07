import sys

sys.path.insert(0, 'evoman')
from controller import Controller


class NEATController(Controller):
    def __init__(self, ff_network=None):
        self.ffNetwork = ff_network
        pass

    def control(self, inputs, controller):
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        output = controller.ffNetwork.activate(inputs)

        # takes decisions about sprite actions
        left = 1 if output[0] > 0 else 0
        right = 1 if output[1] > 0 else 0
        jump = 1 if output[2] > 0 else 0
        shoot = 1 if output[3] > 0 else 0
        release = 1 if output[4] > 0 else 0
        return [left, right, jump, shoot, release]
