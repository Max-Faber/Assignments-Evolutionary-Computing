import os, sys, neat, NEAT_visualize, pickle, numpy

sys.path.insert(0, 'evoman')
from NEAT_evoman_controller import NEATController
from environment import Environment
from archive.demo_controller import player_controller

class EvomanNEAT:

    def __init__(self, neat_config, number_of_gens, experiment_env, enemies, enable_enemy_hint=False):
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

        self.check_points_dir = experiment_env + "/checkpoints"
        if not os.path.exists(self.check_points_dir):
            os.makedirs(self.check_points_dir)

        self.enable_enemy_hint = enable_enemy_hint
        self.experiment_env = experiment_env
        self.neat_config_file = neat_config
        self.number_of_gens = number_of_gens
        self.enemies = enemies
        self.gen = 0
        self.max_fitness = float('-inf')
        self.fitnesses = []
        self.stats = neat.StatisticsReporter()
        self.fitnesses_per_gen = []
        pass

    def eval_genomes(self, genomes, cfg):
        self.gen += 1

        sim = 0
        all_fitness = []
        for _, genome in genomes:
            fitnesses = self.eval_genome(genome, enemies=self.enemies)
            genome.fitness = numpy.mean(list(fitnesses.values())) - numpy.std(list(fitnesses.values()))
            all_fitness.append(genome.fitness)
            if genome.fitness > self.max_fitness and genome.fitness > 0:
                self.max_fitness = genome.fitness
                self.fitnesses = fitnesses
                with open(self.high_scores_dir + '/highest_genome({:.1f}_fitness).pk1'.format(genome.fitness),
                          'wb') as output:
                    pickle.dump(genome, output)
            sim += 1
            print('Generation: {}, simulation: {}'.format(self.gen, sim))
        self.fitnesses_per_gen.append(all_fitness)
        NEAT_visualize.plot_stats(self.stats, ylog=False, view=False, filename=self.graphs_dir + '/avg_fitness.svg')

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
        return numpy.array(weights)

    def eval_genome(self, genome, enemies, env_speed='fastest'):
        # ff_network = neat.nn.FeedForwardNetwork.create(genome, config)
        fitnesses = {}
        # play against all configured enemies with the same genome instance (generalist)
        for e in enemies:
            f, p, _, t = self.make_env_for_enemy(e, env_speed).play(pcont=self.weights_from_genome(genome))
            fitnesses[str(e)] = f
        return fitnesses

    def neat_config(self):
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.neat_config_file)

    def make_env_for_enemy(self, enemy, env_speed='fastest'):
        return Environment(experiment_name=self.experiment_env, speed=env_speed, playermode='ai', enemymode='static',
                           player_controller=player_controller(_n_hidden=10), enemies=[enemy], logs='on',
                           randomini='yes')

    def run(self):
        config = self.neat_config()
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
        return winner, self.fitnesses
