import os, sys, neat, visualize, pickle, numpy as np
from numpy.core.numeric import outer
sys.path.insert(0, 'evoman') 
from environment import Environment
from controller import Controller

experiment_name = 'NEAT'
ffNetwork       = None
stats           = None
highest_fitness = float('-inf')

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
        left    = 1 if output[0] > 0 else 0
        right   = 1 if output[1] > 0 else 0
        jump    = 1 if output[2] > 0 else 0
        shoot   = 1 if output[3] > 0 else 0
        release = 1 if output[4] > 0 else 0
        return [left, right, jump, shoot, release]

generation = 0

def eval_genomes(genomes, config):
    global ffNetwork, generation, stats, npop, highest_fitness
    sim             = 0
    
    generation += 1
    for _, genome in genomes:
        genome.fitness = 0.0
        ffNetwork = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness += simulation(env, None)
        if genome.fitness > highest_fitness and genome.fitness > 0:
            highest_fitness = genome.fitness
            with open('Highscores/highest_genome({}_gain).pk1'.format(highest_fitness), 'wb') as output:
                pickle.dump(genome, output)
        sim += 1
        print('Generation: {}, simulation: {}'.format(generation, sim))
    visualize.plot_stats(stats, ylog=False, view=False, filename='Graphs/avg_fitness.svg')

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
    # f, p, e, t = env.play(pcont=x) # fitness, self.player.life, self.enemy.life, self.time
    f, p, e, t = env.play() # fitness, self.player.life, self.enemy.life, self.time
    # return f
    return p - e

def run(config_file):
    global stats
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

    # playWithWinningGenome(config)
    winner  = p.run(eval_genomes, 10)
    visualize.draw_net(config, winner, True, filename='Graphs/network.svg')
    visualize.plot_stats(stats, ylog=False, view=True, filename='Graphs/avg_fitness.svg')
    visualize.plot_species(stats, view=True, filename='Graphs/spediation.svg')

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    with open('Highscores/winner.pk1', 'wb') as output:
        pickle.dump(winner, output)

def playWithWinningGenome(config):
    global ffNetwork

    with open('Highscores/highest_genome_98_gain.pk1', 'rb') as output:
        genome = pickle.load(output)
    ffNetwork = neat.nn.FeedForwardNetwork.create(genome, config)
    env.speed = 'normal'
    env.play()
    sys.exit()

if __name__ == "__main__":
    local_dir   = os.path.dirname(__file__)
    run(os.path.join(local_dir, "config-feedforward.txt"))