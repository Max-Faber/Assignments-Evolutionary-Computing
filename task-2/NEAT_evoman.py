import os, sys, neat, NEAT_visualize, pickle, numpy

sys.path.insert(0, 'evoman')
from NEAT_evoman_controller import NEATController
from environment import Environment


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
        self.stats = neat.StatisticsReporter()
        self.fitnesses_per_gen = []
        pass

    def eval_genomes(self, genomes, cfg):
        self.gen += 1

        sim = 0
        all_fitness = []
        for _, genome in genomes:
            genome.fitness = self.eval_genome(genome, cfg, enemies=self.enemies)
            all_fitness.append(genome.fitness)
            if genome.fitness > self.max_fitness and genome.fitness > 0:
                self.max_fitness = genome.fitness
                with open(self.high_scores_dir + '/highest_genome({:.1f}_fitness).pk1'.format(genome.fitness),
                          'wb') as output:
                    pickle.dump(genome, output)
            sim += 1
            print('Generation: {}, simulation: {}'.format(self.gen, sim))
        self.fitnesses_per_gen.append(all_fitness)
        NEAT_visualize.plot_stats(self.stats, ylog=False, view=False, filename=self.graphs_dir + '/avg_fitness.svg')

    def eval_genome(self, genome, config, enemies, env_speed='fastest'):
        ff_network = neat.nn.FeedForwardNetwork.create(genome, config)
        fitnesses = []
        # play against all configured enemies with the same genome instance (generalist)
        for e in enemies:
            controller = NEATController(ff_network, enemy_hint=None if not self.enable_enemy_hint else e)
            f, p, e, t = self.make_env_for_enemy(e, env_speed).play(controller)
            fitnesses.append(f)
        return numpy.mean(fitnesses) - numpy.std(fitnesses)

    @staticmethod
    def weights_from_genome(genome):
        weights = []
        for c in genome.connections.values():
            weights.append(c.weight)
        return weights

    def neat_config(self):
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.neat_config_file)

    def make_env_for_enemy(self, enemy, env_speed='fastest'):
        return Environment(experiment_name=self.experiment_env, speed=env_speed, playermode='ai', enemymode='static',
                           player_controller=NEATController(), enemies=[enemy], logs='on', randomini='yes')

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
        return winner, self.eval_genome(winner, self.neat_config(), self.enemies)
