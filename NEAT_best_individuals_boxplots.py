import os, sys, neat, pickle, NEAT_visualize, statistics
from matplotlib.pyplot import box
from time import time, strftime, localtime
from NEAT_evoman_controller import NEATController

sys.path.insert(0, 'evoman')
from environment import Environment

class ExperimentNEAT:
    def __init__(self, base_path, enemy, name):
        self.base_path      = base_path
        self.enemy          = enemy
        self.name           = name
        self.config_path    = f'{self.base_path}/neat-config.txt'
        self.enemy_path     = f'{self.base_path}/enemy-{self.enemy}'
        self.genomes        = []
        self.neat_config    = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                          self.config_path)
        self.env            = Environment(speed='fastest', playermode='ai', enemymode='static',
                                          player_controller=NEATController(), enemies=[enemy], logs='off', randomini='yes')
        self.load_genomes()

    def load_genomes(self):
        for path in os.listdir(self.enemy_path):
            full_path = f'{self.enemy_path}/{path}'

            if not os.path.isdir(full_path) or not path.startswith('round-'):
                continue
            full_path = f'{full_path}/highscores'
            genomes_sorted = sorted(os.listdir(full_path)) # We sort alphabetically so that the last element is the one with the highest individual gain
            with open(f'{full_path}/{genomes_sorted[len(genomes_sorted) - 1]}', 'rb') as output:
                self.genomes.append(pickle.load(output))

    def eval_genomes(self, test_quantity):
        mean_fitnesses = []
        for genome in self.genomes:
            mean_fitnesses.append(self.eval_genome(genome, self.neat_config, test_quantity))
        return mean_fitnesses

    def eval_genome(self, genome, config, test_quantity, env_speed='fastest'):
        ff_network      = neat.nn.FeedForwardNetwork.create(genome, config)
        self.env.speed  = env_speed
        neat_controller = NEATController(ff_network)
        fitnesses       = []

        for _ in range(test_quantity):
            f, p, e, t = self.env.play(neat_controller)
            fitnesses.append(p - e)
        return sum(fitnesses) / len(fitnesses)

class IndividualComparer:
    def __init__(self, enemy, individuals_to_compare, results_base_path, test_quantity):
        self.enemy                  = enemy
        self.individuals_to_compare = individuals_to_compare
        self.results_base_path      = results_base_path
        self.test_quantity          = test_quantity
        self.algorithm_names        = []
        for individual_to_compare in self.individuals_to_compare:
            self.algorithm_names.append(individual_to_compare.name)
        self.output_dir = f'{self.results_base_path}/{"_vs_".join(self.algorithm_names)}/enemy-{self.enemy}/{strftime("%d_%m_%Y_%H_%M_%S", localtime(time()))}'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def compare_individuals(self):
        mean_fitnesses_per_individual   = []
        box_names                       = []

        for individual_to_compare in self.individuals_to_compare:
            mean_fitnesses_per_individual.append(individual_to_compare.eval_genomes(test_quantity))
            box_names.append(individual_to_compare.name)
        NEAT_visualize.plot_individuals_avg_fitness(fitnesses_two_dimensional=mean_fitnesses_per_individual,
                                                    box_names=box_names,
                                                    filename=f'{self.output_dir}/best_individuals_enemy-{self.enemy}_boxplot.svg')
        with open(f'{self.output_dir}/statistics.json', 'w') as output:
            stats = []
            for i in range(len(self.individuals_to_compare)):
                stats.append({
                    "name": self.individuals_to_compare[i].name,
                    "std_of_means": statistics.stdev(mean_fitnesses_per_individual[i]),
                    "mean_of_means": sum(mean_fitnesses_per_individual[i]) / len(mean_fitnesses_per_individual[i])
                })
            output.write(str(stats).replace('\'', '\"'))
    
    def eval_genome(self, genome, config, env_speed='fastest'):
        with open(self.path, 'rb') as output:
            genome = pickle.load(output)
        ff_network = neat.nn.FeedForwardNetwork.create(genome, config)
        self.env.speed = env_speed
        f, p, e, t = self.env.play(NEATController(ff_network))
        return p - e  # individual gain as fitness

if __name__ == '__main__':
    final_results           = 'NEAT-final-results'
    results_base_path       = f'{final_results}/Comparisons'
    enemy                   = 2
    test_quantity           = 5 # Number of games of which the mean will be taken
    individuals_to_compare  = []

    # Enemy 2
    individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/EA1/NEAT-v1-23_09_2021_18_47_15',
                                                 enemy=enemy,
                                                 name='NEAT-v1'))
    individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/EA2/NEAT-v2-27_09_2021_21_25_11',
                                                 enemy=enemy,
                                                 name='NEAT-v2'))

    # Enemy 8
    # individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/EA1/NEAT-v1-23_09_2021_23_48_20',
    #                                              enemy=enemy,
    #                                              name='NEAT-v1'))
    # individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/EA2/NEAT-v2-27_09_2021_20_55_36',
    #                                              enemy=enemy,
    #                                              name='NEAT-v2'))

    IndividualComparer(enemy=enemy,
                       individuals_to_compare=individuals_to_compare,
                       results_base_path=results_base_path,
                       test_quantity=test_quantity).compare_individuals()