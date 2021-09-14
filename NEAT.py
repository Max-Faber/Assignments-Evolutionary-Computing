import os, sys, neat, inspect, pickle, numpy as np
from numpy.core.numeric import outer
sys.path.insert(0, 'evoman') 
from environment import Environment
from controller import Controller

experiment_name = 'NEAT'
ffNetwork       = None

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Source: https://neat-python.readthedocs.io/en/latest/xor_example.html#example-source

# implements controller structure for player
class player_controller(Controller):
    def __init__(self):
        pass

    def control(self, inputs, controller):
        global ffNetwork

        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        output = ffNetwork.activate(inputs)
        # takes decisions about sprite actions 
        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]

generation = 0

def eval_genomes(genomes, config):
    global ffNetwork, generation
    nets    = []
    sim     = 0


    for genome_id, genome in genomes:
        genome.fitness = 0.0
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
    fit_pop = []
    for index, y in enumerate(pop):
        ffNetwork = nets[index]
        fit_pop.append(simulation(env, y))
        genomes[index][1].fitness += fit_pop[len(fit_pop) - 1]
        print('Generation: {}, simulation: {}'.format(generation, sim))
        sim += 1
    generation += 1
    # f, p, e, t = env.play(pcont=...) # fitness, self.player.life, self.enemy.life, self.time

env                 = Environment(experiment_name=experiment_name, speed='fastest', playermode='ai',
                                  player_controller=player_controller(), enemymode='static',
                                  enemies=[2], logs='on')
# number of weights for multilayer with 10 hidden neurons.
n_hidden_neurons    = 10
n_vars              = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
dom_l               = -1
dom_u               = 1
npop                = 50
pop                 = np.random.uniform(dom_l, dom_u, (npop, n_vars))

# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x) # fitness, self.player.life, self.enemy.life, self.time
    return f

def run(config_file):
    config   = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_file)
    p        = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-111')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats    = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    playWithWinningGenome(config)
    # winner  = p.run(eval_genomes, 50)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    with open('winner.pk1', 'wb') as output:
        pickle.dump(winner, output)

def playWithWinningGenome(config):
    global ffNetwork

    with open('winner.pk1', 'rb') as output:
        genome = pickle.load(output)
    ffNetwork = neat.nn.FeedForwardNetwork.create(genome, config)
    env.speed = 'normal'
    env.play()
    sys.exit()

if __name__ == "__main__":
    local_dir   = os.path.dirname(__file__)
    run(os.path.join(local_dir, "config-feedforward.txt"))