import sys, shutil, os, pickle, numpy

sys.path.insert(0, 'evoman')
from time import time, strftime, localtime
from NEAT_evoman import EvomanNEAT
import NEAT_visualize

number_of_rounds = 10
experiments = [
    {
        "name": "NEAT-v1",
        "neat-config-file": "NEAT-configs/config-feedforward-1.txt",
        "enemies": [1],
        "number-of-generations": 30,
        "best-genome-test-quantity": 5
    }#,
    # {
    #     "name": "NEAT-v1",
    #     "neat-config-file": "NEAT-configs",
    #     "enemies": [2],
    #     "number-of-generations": 30,
    #     "best-genome-test-quantity": 5
    # },
    # {
    #     "name": "NEAT-v1",
    #     "neat-config-file": "NEAT-configs",
    #     "enemies": [3],
    #     "number-of-generations": 30,
    #     "best-genome-test-quantity": 5
    # },
    # {
    #     "name": "NEAT-v2",
    #     "config-file": "NEAT-configs/config-feedforward-2.txt",
    #     "3-enemies": [1, 2, 3],
    #     "number-of-generations": 25
    # }
]


class Experiment:

    def __init__(self, cfg):
        self.name = "NEAT-results/" + cfg["name"]
        self.neat_cfg = cfg["neat-config-file"]
        self.enemies = cfg["enemies"]
        self.num_gens = cfg["number-of-generations"]
        self.base_env = self.name + '-' + strftime("%d_%m_%Y_%H_%M_%S", localtime(time()))
        self.best_genome_test_qt = cfg["best-genome-test-quantity"]
        self.graphs_dir = self.base_env + '/graphs'
        os.makedirs(self.base_env)
        shutil.copyfile(self.neat_cfg, self.base_env + '/neat-config.txt')
        pass

    def run(self):
        print("Running experiment: {}, config file used: {}".format(self.name, self.neat_cfg))
        for enemy in self.enemies:
            enemy_env = self.base_env + "/enemy-" + str(enemy)
            os.makedirs(enemy_env)
            enemy_graphs_env = enemy_env + "/graphs"
            os.makedirs(enemy_graphs_env)
            # per round the max per gen [[10, 11, 9, ..., n-rounds], [...], ..., n-gens]
            max_fitness_per_gen = [[] for _ in range(self.num_gens)]
            # per round the avg per gen [[10.4, 11.9, 9.9, ..., n-rounds], [...], ..., n-gens]
            avg_fitness_per_gen = [[] for _ in range(self.num_gens)]
            best_individual_mean_fitnesses = []  # = avg fitness best genome each round -> [80.2, 72.3, ... n-rounds]
            winner_of_winners = { "genome": None, "fitness": -100 }
            for i in range(number_of_rounds):
                enemy_round_env = enemy_env + "/round-" + str(i + 1)

                n = EvomanNEAT(neat_config=self.neat_cfg,
                               number_of_gens=int(self.num_gens),
                               experiment_env=enemy_round_env,
                               enemy=enemy)

                winner, fitness = n.run()
                if fitness > winner_of_winners["fitness"]:
                    winner_of_winners["fitness"] = fitness
                    winner_of_winners["genome"] = winner
                fitnesses = []
                for _ in range(self.best_genome_test_qt):
                    fitnesses.append(n.eval_genome(winner, n.neat_config()))

                best_individual_mean_fitnesses.append(sum(fitnesses) / len(fitnesses))

                for j in range(len(n.fitnesses_per_gen)):
                    gen_fitness = n.fitnesses_per_gen[j]
                    max_fitness_per_gen[j].append(max(gen_fitness))
                    avg_fitness_per_gen[j].append(sum(gen_fitness) / len(gen_fitness))
                
            with open('{}/winner_of_winners(gain_{}).pk1'.format(enemy_env, winner_of_winners['fitness']), 'wb') as output:
                pickle.dump(winner_of_winners["genome"], output)

            NEAT_visualize.plot_individual_avg_fitness(self.name,
                                                       best_individual_mean_fitnesses,
                                                       filename=enemy_graphs_env + '/best_individual_avg_fitness.svg')
            NEAT_visualize.plot_fitnesses(avg_fitness_per_gen, max_fitness_per_gen,
                                          filename=enemy_graphs_env + '/gen_fitnesses.svg')


if __name__ == '__main__':
    for experiment in experiments:
        Experiment(experiment).run()
