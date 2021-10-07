import os, sys, neat, NEAT_visualize, pickle

sys.path.insert(0, 'evoman')
from NEAT_evoman_controller import NEATController
from environment import Environment


class EvomanNEAT:

    def __init__(self, neat_config, number_of_gens, experiment_env, enemy):
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

        self.experiment_env = experiment_env
        self.neat_config_file = neat_config
        self.number_of_gens = number_of_gens
        self.gen = 0
        self.env = Environment(experiment_name=experiment_env, speed='fastest', playermode='ai', enemymode='static',
                               player_controller=NEATController(), enemies=[enemy], logs='on', randomini='yes')
        self.max_fitness = float('-inf')
        self.stats = neat.StatisticsReporter()

        self.fitnesses_per_gen = []
        pass

    def eval_genomes(self, genomes, cfg):
        self.gen += 1

        sim = 0
        all_fitness = []
        for _, genome in genomes:
            genome.fitness = 0.0
            genome.fitness += self.eval_genome(genome, cfg)
            all_fitness.append(genome.fitness)
            if genome.fitness > self.max_fitness and genome.fitness > 0:
                self.max_fitness = genome.fitness
                with open(self.high_scores_dir + '/highest_genome({}_gain).pk1'.format(genome.fitness),
                          'wb') as output:
                    pickle.dump(genome, output)
            sim += 1
            print('Generation: {}, simulation: {}'.format(self.gen, sim))
        self.fitnesses_per_gen.append(all_fitness)
        NEAT_visualize.plot_stats(self.stats, ylog=False, view=False, filename=self.graphs_dir + '/avg_fitness.svg')

    def eval_genome(self, genome, config, env_speed='fastest'):
        ff_network = neat.nn.FeedForwardNetwork.create(genome, config)
        self.env.speed = env_speed
        f, p, e, t = self.env.play(NEATController(ff_network))
        return p - e  # individual gain as fitness

    def neat_config(self):
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.neat_config_file)

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
        return winner, self.eval_genome(winner, self.neat_config())
