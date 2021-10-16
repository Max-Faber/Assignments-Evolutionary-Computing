import sys, numpy as np

sys.path.insert(0, 'evoman')
from controller import Controller

class NEATController(Controller):
    def __init__(self, enemy_hint):
        self.enemy_hint = enemy_hint
        self.n_hidden = [10]
        pass

    @staticmethod
    def sigmoid_activation(x):
        return 1. / (1. + np.exp(-x))

    def control(self, inputs, controller):
        inputs = np.append(inputs, self.enemy_hint)
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        # Preparing the weights and biases from the controller of layer 1

        # Biases for the n hidden neurons
        bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
        # Weights for the connections from the inputs to the hidden nodes
        weights1_slice = len(inputs) * self.n_hidden[0] + self.n_hidden[0]
        weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs), self.n_hidden[0]))

        # Outputs activation first layer.
        output1 = NEATController.sigmoid_activation(inputs.dot(weights1) + bias1)

        # Preparing the weights and biases from the controller of layer 2
        bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
        weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

        # Outputting activated second layer. Each entry in the output is an action
        output = NEATController.sigmoid_activation(output1.dot(weights2) + bias2)[0]

        # takes decisions about sprite actions (sigmoid activations so 0.5 boundary)
        left = 1 if output[0] > 0.5 else 0
        right = 1 if output[1] > 0.5 else 0
        jump = 1 if output[2] > 0.5 else 0
        shoot = 1 if output[3] > 0.5 else 0
        release = 1 if output[4] > 0.5 else 0
        return [left, right, jump, shoot, release]
