import os, sys, neat, pickle, NEAT_visualize, statistics, re
from scipy import stats
from time import time, strftime, localtime
from NEAT_evoman_controller import NEATController
from NEAT_evoman import EvomanNEAT
from archive.demo_controller import player_controller

sys.path.insert(0, 'evoman')
from environment import Environment

class ExperimentNEAT:
    def __init__(self, base_path, enemies, name, enable_enemy_hint):
        self.base_path          = base_path
        self.enemies            = enemies
        self.name               = name
        self.enable_enemy_hint  = enable_enemy_hint
        self.genomes            = []
        self.controller         = player_controller(_n_hidden=10) if not self.enable_enemy_hint\
            else NEATController(self.enable_enemy_hint)
        self.load_genomes()

    @staticmethod
    def get_fitness_regex(search_text):
        fitness = re.findall('gen\d+_genome\d+\(?([-+][0-9]*\.[0-9]+|[0-9]*\.[0-9]+)_fitness\).pk1', search_text)
        return -1.0 if len(fitness) == 0 else float(fitness[0])

    def load_genomes(self):
        for path in os.listdir(self.base_path):
            full_path = f'{self.base_path}/{path}'

            if not os.path.isdir(full_path) or not path.startswith('round-'):
                continue
            full_path       = f'{full_path}/highscores'
            pickle_dumps    = list(filter(lambda x: x.endswith('.pk1'), os.listdir(full_path)))
            genomes_sorted  = sorted(pickle_dumps, key=self.get_fitness_regex) # We sort the fitness ascending so that the last element is the one with the highest individual gain
            with open(f'{full_path}/{genomes_sorted[len(genomes_sorted) - 1]}', 'rb') as output:
                self.genomes.append(pickle.load(output))

    def eval_genomes(self, test_quantity, name):
        mean_fitnesses = []
        for i, genome in enumerate(self.genomes):
            mean_fitnesses.append(self.eval_genome(genome, test_quantity, name, i, len(self.genomes)))
        return mean_fitnesses

    def get_env(self, enemy):
        return Environment(speed='fastest', playermode='ai', enemymode='static',
                           player_controller=self.controller, enemies=[enemy],
                           logs='off', randomini='yes')

    def eval_genome(self, genome, test_quantity, name, genome_index, genomes_length):
        fitnesses       = []

        for tq in range(test_quantity):
            fitnesses_enemies = []
            for enemy in self.enemies:
                f, p, e, t = self.get_env(enemy).play(pcont=EvomanNEAT.weights_from_genome(genome, self.enable_enemy_hint))
                print(f'EA: {name}, genome: {genome_index + 1}/{genomes_length}, test: {tq + 1}/{test_quantity}, enemy: {enemy}')
                fitnesses_enemies.append(p - e)
            fitnesses.append(sum(fitnesses_enemies) / len(fitnesses_enemies))
        return sum(fitnesses) / len(fitnesses) # Calculate the mean of fitnesses

class IndividualComparer:
    def __init__(self, enemies, individuals_to_compare, results_base_path, test_quantity):
        self.enemies                = enemies
        self.individuals_to_compare = individuals_to_compare
        self.results_base_path      = results_base_path
        self.test_quantity          = test_quantity
        self.algorithm_names        = []
        self.algorithm_names        = [individual_to_compare.name
                                       for individual_to_compare in self.individuals_to_compare]
        self.output_dir = f'{self.results_base_path}/{"_vs_".join(self.algorithm_names)}/' \
                          f'{strftime("%d_%m_%Y_%H_%M_%S", localtime(time()))}'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def compare_individuals(self):
        mean_fitnesses_per_individual   = []
        box_names                       = []

        if len(self.individuals_to_compare) != 2:
            return # Otherwise we can't perform a statistical t-test
        for individual_to_compare in self.individuals_to_compare:
            mean_fitnesses_per_individual.append(individual_to_compare.eval_genomes(test_quantity, individual_to_compare.name))
            box_names.append(individual_to_compare.name)
        NEAT_visualize.plot_individuals_avg_fitness(fitnesses_two_dimensional=mean_fitnesses_per_individual,
                                                    box_names=box_names,
                                                    filename=f'{self.output_dir}/best_individuals_enemies-'
                                                             f'{self.enemies}_boxplot.svg')
        t_test = stats.ttest_ind(mean_fitnesses_per_individual[0], mean_fitnesses_per_individual[1])
        with open(f'{self.output_dir}/statistics.json', 'w') as output:
            comapre_stats = []
            for individual_to_compare, mean_fitness in zip(self.individuals_to_compare, mean_fitnesses_per_individual):
                comapre_stats.append({
                    "name": individual_to_compare.name,
                    "enemies": self.enemies,
                    "std_of_means": statistics.stdev(mean_fitness),
                    "mean_of_means": sum(mean_fitness) / len(mean_fitness),
                    "independent_t_test": {
                        "statistic": t_test.statistic,
                        "pvalue": t_test.pvalue
                    }
                })
            output.write(str(comapre_stats).replace('\'', '\"'))

if __name__ == '__main__':
    final_results           = 'NEAT-final-results'
    results_base_path       = f'{final_results}/Comparisons'
    enemies                 = [1, 2, 3, 4, 5, 6, 7, 8]
    test_quantity           = 5 # Number of games of which the mean will be taken
    individuals_to_compare  = []

    # Enemies 1, 2, 3
    # individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/v1/NEAT-v1-14_10_2021_17_08_32',
    #                                              enemies=enemies,
    #                                              name='NEAT-v1_enemies_1_2_3',
    #                                              enable_enemy_hint=False))
    # individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/v2/NEAT-v2-16_10_2021_08_44_36',
    #                                              enemies=enemies,
    #                                              name='NEAT-v2_enemies_1_2_3',
    #                                              enable_enemy_hint=True))

    # Enemies 4, 5, 8
    individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/v1/NEAT-v1-16_10_2021_01_28_09',
                                                 enemies=enemies,
                                                 name='NEAT-v1_enemies_4_5_8',
                                                 enable_enemy_hint=False))
    individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/v2/NEAT-v2-16_10_2021_01_27_33',
                                                 enemies=enemies,
                                                 name='NEAT-v2_enemies_4_5_8',
                                                 enable_enemy_hint=True))

    # Enemies 6, 7, 8
    # individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/v1/NEAT-v1-15_10_2021_09_01_39',
    #                                              enemies=enemies,
    #                                              name='NEAT-v1_enemies_6_7_8',
    #                                              enable_enemy_hint=False))
    # individuals_to_compare.append(ExperimentNEAT(base_path=f'{final_results}/v2/NEAT-v2-15_10_2021_17_34_09',
    #                                              enemies=enemies,
    #                                              name='NEAT-v2_enemies_6_7_8',
    #                                              enable_enemy_hint=True))

    IndividualComparer(enemies=enemies,
                       individuals_to_compare=individuals_to_compare,
                       results_base_path=results_base_path,
                       test_quantity=test_quantity).compare_individuals()