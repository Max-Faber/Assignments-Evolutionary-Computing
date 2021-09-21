import os, sys, neat, visualize, pickle

sys.path.insert(0, 'evoman')
from controller import Controller
from environment import Environment
from time import time, strftime, localtime


class NEATController(Controller):
    def __init__(self, ff_network):
        self.ffNetwork = ff_network
        pass

    def control(self, inputs):
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
        output = self.ffNetwork.activate(inputs)

        # takes decisions about sprite actions
        left = 1 if output[0] > 0 else 0
        right = 1 if output[1] > 0 else 0
        jump = 1 if output[2] > 0 else 0
        shoot = 1 if output[3] > 0 else 0
        release = 1 if output[4] > 0 else 0
        return [left, right, jump, shoot, release]


class EvomanNEAT:

    def __init__(self, neat_config, experiment_env, enemy):
        if not os.path.exists(experiment_env):
            os.makedirs(experiment_env)
        winner_dir = experiment_env + "/winner"
        if not os.path.exists(winner_dir):
            os.makedirs(winner_dir)
        self.winner_file = winner_dir + "/winner.pk1"
        self.graphs_dir = experiment_env + "/graphs"
        if not os.path.exists(self.graphs_dir):
            os.makedirs(self.graphs_dir)
        self.high_scores_dir = experiment_env + "/highscores"
        if not os.path.exists(self.high_scores_dir):
            os.makedirs(self.high_scores_dir)
        self.experiment_env = experiment_env
        self.neat_config = neat_config
        self.gen = 0
        self.n_pop = 50
        self.env = Environment(experiment_name=experiment_env,
                               speed='fastest',
                               playermode='ai',
                               enemymode='static',
                               enemies=[enemy],
                               logs='on')
        self.max_fitness = float('-inf')
        self.stats = neat.StatisticsReporter()
        pass

    def eval_genomes(self, genomes, config):
        sim = 0

        self.gen += 1
        for _, genome in genomes:
            genome.fitness = 0.0
            ff_network = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness += self.evaluate(ff_network=ff_network)
            if genome.fitness > self.max_fitness and genome.fitness > 0:
                highest_fitness = genome.fitness
                with open(self.high_scores_dir + '/highest_genome({}_gain).pk1'.format(highest_fitness),
                          'wb') as output:
                    pickle.dump(genome, output)
            sim += 1
            print('Generation: {}, simulation: {}'.format(self.gen, sim))
        visualize.plot_stats(self.stats, ylog=False, view=False, filename=self.graphs_dir + '/avg_fitness.svg')

    # runs simulation
    def evaluate(self, ff_network):
        f, p, e, t = self.env.play(NEATController(ff_network))  # fitness, self.player.life, self.enemy.life, self.time
        return p - e  # individual gain used as fitness

    def run(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.neat_config)
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(self.stats)
        p.add_reporter(neat.Checkpointer(5))

        winner = p.run(self.eval_genomes, 10)
        visualize.draw_net(config, winner, view=False, filename=self.graphs_dir + '/network.svg')
        visualize.plot_stats(self.stats, ylog=False, view=False, filename=self.graphs_dir + '/avg_fitness.svg')
        visualize.plot_species(self.stats, view=False, filename=self.graphs_dir + '/spediation.svg')

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))
        with open(self.winner_file, 'wb') as output:
            pickle.dump(winner, output)
        return winner

    def play_winning_genome(self, winner_file=None):
        if winner_file is None:
            winner_file = self.winner_file
        with open(winner_file, 'rb') as output:
            genome = pickle.load(output)
        ff_network = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
        self.env.speed = 'normal'
        self.evaluate(ff_network)
        sys.exit()


experiments = [
    {
        "name": "NEAT-v1",
        "config-file": "NEAT-configs/config-feedforward-1.txt",
        "3-enemies": [1, 2, 3]
    },
    {
        "name": "NEAT-v2",
        "config-file": "NEAT-configs/config-feedforward-2.txt",
        "3-enemies": [1, 2, 3]
    }
]

if __name__ == '__main__':
    dt = strftime("%d_%m_%Y_%H_%M_%S", localtime(time()))
    for experiment in experiments:
        name = experiment["name"]
        config = experiment["config-file"]
        print("Running experiment: {}, config file used: {}".format(name, config))
        for enemy in experiment["3-enemies"]:
            env_dir = "NEAT-results/" + name + "/" + dt + "/enemy-" + str(enemy)
            n = EvomanNEAT(experiment["config-file"], env_dir, enemy)
            winner = n.run()
