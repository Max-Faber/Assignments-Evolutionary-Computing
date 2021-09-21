import os, sys, neat, NEAT_visualize, pickle
import shutil

sys.path.insert(0, 'evoman')
from NEAT_evoman_controller import NEATController
from shutil import copyfile
from environment import Environment


class EvomanNEAT:

    def __init__(self, neat_config, number_of_gens, experiment_env, enemy):
        if not os.path.exists(experiment_env):
            os.makedirs(experiment_env)

        shutil.copyfile(neat_config, experiment_env + '/neat-config.txt')

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

        self.check_points_dir = experiment_env + "/checkpoints"
        if not os.path.exists(self.check_points_dir):
            os.makedirs(self.check_points_dir)

        self.experiment_env = experiment_env
        self.neat_config = neat_config
        self.number_of_gens = number_of_gens
        self.gen = 0
        self.n_pop = 50
        self.env = Environment(experiment_name=experiment_env,
                               speed='fastest',
                               playermode='ai',
                               enemymode='static',
                               player_controller=NEATController(),
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
                self.max_fitness = genome.fitness
                with open(self.high_scores_dir + '/highest_genome({}_gain).pk1'.format(genome.fitness),
                          'wb') as output:
                    pickle.dump(genome, output)
            sim += 1
            print('Generation: {}, simulation: {}'.format(self.gen, sim))
        NEAT_visualize.plot_stats(self.stats, ylog=False, view=False, filename=self.graphs_dir + '/avg_fitness.svg')

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
        p.add_reporter(
            neat.Checkpointer(min(self.number_of_gens, 5), filename_prefix=self.check_points_dir + '/checkpoint-'))

        winner = p.run(self.eval_genomes, self.number_of_gens)
        NEAT_visualize.draw_net(config, winner, view=False, filename=self.graphs_dir + '/network.svg')
        NEAT_visualize.plot_stats(self.stats, ylog=False, view=False, filename=self.graphs_dir + '/avg_fitness.svg')
        NEAT_visualize.plot_species(self.stats, view=False, filename=self.graphs_dir + '/speciation.svg')

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
