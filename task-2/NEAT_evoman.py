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

    def eval_genomes(self, genomes, _):
        self.gen += 1

        sim = 0
        all_fitness = []
        for _, genome in genomes:
            ind_gains = self.eval_genome(genome, self.enemies)
            num_wins = sum(i > 0 for i in ind_gains.values())
            genome.fitness = self.summarize_eval(ind_gains)
            all_fitness.append(genome.fitness)
            if genome.fitness > self.max_fitness:
                self.max_fitness = genome.fitness
                self.fitnesses = ind_gains
                if not self.enable_enemy_hint:
                    numpy.savetxt(self.high_scores_dir + f'/gen{self.gen}_genome{genome.key}_weights.txt',
                                  self.weights_from_genome(genome))
                with open(self.high_scores_dir + f'/gen{self.gen}_genome{genome.key}_ind_gains.json', 'w') as output:
                    output.write(str(ind_gains).replace('\'', '\"'))
                with open(self.high_scores_dir + '/gen{}_genome{}({:.1f}_fitness).pk1'.format(self.gen, genome.key,
                                                                                              genome.fitness),
                          'wb') as output:
                    pickle.dump(genome, output)
            sim += 1
            print(
                'Generation: {}, simulation: {}, fitness: {}, wins: {}'.format(self.gen, sim, genome.fitness, num_wins))
        self.fitnesses_per_gen.append(all_fitness)
        NEAT_visualize.plot_stats(self.stats, ylog=False, view=False, filename=self.graphs_dir + '/avg_fitness.svg')

    # NN structure, based on control() in demo_controller.py:
    # bias1:    0   t/m 9   ->  10 weights
    # weights1: 10  t/m 209 -> 200 weights
    # bias2:    210 t/m 214 ->   5 weights
    # weights2: 215 t/m 264 ->  50 weights
    @staticmethod
    def weights_from_genome(genome, enemy_hint = False):
        extra_wheights = 10 if enemy_hint else 0
        weights = [None] * (265 + extra_wheights)  # NN has 265 weights in total

        for i, n in enumerate(genome.nodes.values()):
            weights[i if i <= 9 else i + (210 + extra_wheights) - 10] = n.bias
        for i, c in enumerate(genome.connections.values()):
            weights[(i + 10) if (i + 10) <= (209 + extra_wheights) else (i + 10) + 5] = c.weight
        return numpy.array(weights)

    def eval_genome(self, genome, enemies, env_speed='fastest'):
        ind_gains = {}
        # play against all configured enemies with the same genome instance (generalist)
        for e in enemies:
            controller = player_controller(_n_hidden=10) if not self.enable_enemy_hint \
                else NEATController(self.enable_enemy_hint)
            f, p_energy, e_energy, t = self.make_env_for_enemy(e, controller, env_speed).play(
                pcont=self.weights_from_genome(genome, self.enable_enemy_hint))
            ind_gains[str(e)] = p_energy - e_energy
        return ind_gains

    @staticmethod
    def summarize_eval(individual_gains):
        won_games = sum(i > 0 for i in individual_gains.values())
        return won_games + (sum(list(individual_gains.values())) / 100)

    def neat_config(self):
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.neat_config_file)

    def make_env_for_enemy(self, enemy, controller, env_speed='fastest'):
        return Environment(experiment_name=self.experiment_env, speed=env_speed, playermode='ai', enemymode='static',
                           player_controller=controller, enemies=[enemy], logs='off',
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
