import sys
import numpy as np

sys.path.insert(0, 'evoman')
from controller import Controller


class NEATController(Controller):
    def __init__(self, genome=None, enemy_hint=None):
        self.genome = genome
        self.enemy_hint = enemy_hint
        self.output_threshold = 0.5
        self.n_hidden = [10]
        pass

    @staticmethod
    def sigmoid_activation(x):
        return 1. / (1. + np.exp(-x))

    # NN structure, based on control() in demo_controller.py:
    # bias1:    0   t/m 9   ->  10 weights
    # weights1: 10  t/m 209 -> 200 weights
    # bias2:    210 t/m 214 ->   5 weights
    # weights2: 215 t/m 264 ->  50 weights
    @staticmethod
    def weights_from_genome(genome):
        weights = [None] * 265  # NN has 265 weights in total

        for i, n in enumerate(genome.nodes.values()):
            weights[i if i <= 9 else i + 210 - 10] = n.bias
        for i, c in enumerate(genome.connections.values()):
            weights[(i + 10) if (i + 10) <= 209 else (i + 10) + 5] = c.weight
        return np.array(weights)

    def control(self, inputs, controller):
        if controller.enemy_hint is not None:
            inputs.insert(0, controller.enemy_hint)
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        # output = controller.ffNetwork.activate(inputs)
        weights = controller.weights_from_genome(controller.genome)

        bias1 = weights[:controller.n_hidden[0]].reshape(1, controller.n_hidden[0])
        # Weights for the connections from the inputs to the hidden nodes
        weights1_slice = len(inputs) * controller.n_hidden[0] + controller.n_hidden[0]
        weights1 = weights[controller.n_hidden[0]:weights1_slice].reshape((len(inputs), controller.n_hidden[0]))

        # Outputs activation first layer.
        output1 = controller.sigmoid_activation(inputs.dot(weights1) + bias1)

        # Preparing the weights and biases from the controller of layer 2
        bias2 = weights[weights1_slice:weights1_slice + 5].reshape(1, 5)
        weights2 = weights[weights1_slice + 5:].reshape((controller.n_hidden[0], 5))

        # Outputting activated second layer. Each entry in the output is an action
        output = controller.sigmoid_activation(output1.dot(weights2) + bias2)[0]

        # takes decisions about sprite actions
        left = 1 if output[0] > self.output_threshold else 0
        right = 1 if output[1] > self.output_threshold else 0
        jump = 1 if output[2] > self.output_threshold else 0
        shoot = 1 if output[3] > self.output_threshold else 0
        release = 1 if output[4] > self.output_threshold else 0
        return [left, right, jump, shoot, release]
