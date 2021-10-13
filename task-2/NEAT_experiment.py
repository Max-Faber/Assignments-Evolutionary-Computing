import sys, shutil, os, pickle, numpy, warnings

sys.path.insert(0, 'evoman')
from time import time, strftime, localtime
from NEAT_evoman import EvomanNEAT
from NEAT_evoman_controller import NEATController
import NEAT_visualize

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
number_of_rounds = 10
experiments = [
    {
        "name": "NEAT-v1",
        "neat-config-file": "NEAT-configs/config-feedforward-1.txt",
        "enemies": [1, 2, 3],
        "number-of-generations": 50,
        "best-genome-test-quantity": 5,
        "enable-enemy-hint": False
    }
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
        self.enable_enemy_hint = cfg["enable-enemy-hint"]
        os.makedirs(self.base_env)
        numpy.savetxt(self.base_env + "/enemies.txt", self.enemies, '%s', delimiter=',')
        shutil.copyfile(self.neat_cfg, self.base_env + '/neat-config.txt')
        pass

    def run(self):
        print("Running experiment: {}, config file used: {}".format(self.name, self.neat_cfg))
        enemy_graphs_env = self.base_env + "/graphs"
        os.makedirs(enemy_graphs_env)
        # per round the max per gen [[10, 11, 9, ..., n-rounds], [...], ..., n-gens]
        max_fitness_per_gen = [[] for _ in range(self.num_gens)]
        # per round the avg per gen [[10.4, 11.9, 9.9, ..., n-rounds], [...], ..., n-gens]
        avg_fitness_per_gen = [[] for _ in range(self.num_gens)]
        best_individual_mean_fitnesses = []  # = avg fitness best genome each round -> [80.2, 72.3, ... n-rounds]
        winner_of_winners = {"genome": None, "fitness": -100, "enemie_fitnesses": {}}
        for i in range(number_of_rounds):
            enemy_round_env = self.base_env + "/round-" + str(i + 1)

            n = EvomanNEAT(neat_config=self.neat_cfg,
                           number_of_gens=int(self.num_gens),
                           experiment_env=enemy_round_env,
                           enemies=self.enemies,
                           enable_enemy_hint=self.enable_enemy_hint)

            winner, fitness = n.run()
            summ_fitness = n.summarize_eval(fitness)
            if summ_fitness > winner_of_winners["fitness"]:
                winner_of_winners["fitness"] = summ_fitness
                winner_of_winners["genome"] = winner
                winner_of_winners["enemie_fitnesses"] = fitness
            fitnesses = []
            for _ in range(self.best_genome_test_qt):
                ind_gains = n.eval_genome(winner, self.enemies)
                fitnesses.append(n.summarize_eval(ind_gains))

            best_individual_mean_fitnesses.append(sum(fitnesses) / len(fitnesses))

            for j in range(len(n.fitnesses_per_gen)):
                gen_fitness = n.fitnesses_per_gen[j]
                max_fitness_per_gen[j].append(max(gen_fitness))
                avg_fitness_per_gen[j].append(sum(gen_fitness) / len(gen_fitness))

        winner_name = '{}/winner_of_winners(fitness_{:.1f})'.format(self.base_env, winner_of_winners['fitness'])
        with open(winner_name + ".pk1", 'wb') as output:
            pickle.dump(winner_of_winners["genome"], output)

        numpy.savetxt(winner_name + 'weights.txt', NEATController.weights_from_genome(winner_of_winners["genome"]))
        with open(winner_name + 'enemie_fitnesses.json', 'w') as output:
            output.write(str(winner_of_winners["enemie_fitnesses"]).replace('\'', '\"'))

        NEAT_visualize.plot_individual_avg_fitness(self.name,
                                                   best_individual_mean_fitnesses,
                                                   filename=enemy_graphs_env + '/best_individual_avg_fitness.svg')
        NEAT_visualize.plot_fitnesses(avg_fitness_per_gen, max_fitness_per_gen,
                                      filename=enemy_graphs_env + '/gen_fitnesses.svg')


if __name__ == '__main__':
    args = sys.argv
    print(args)
    if len(args) == 0:
        warnings.warn("No experiment names were given. Assuming all experiments have to be performed...")
        warnings.warn("You can run specific experiments by calling 'python [name_of_file.py] exp_1 exp_2 exp_3'.")
    for experiment in experiments:
        if args.__contains__(experiment["name"]) or len(args) == 0:
            Experiment(experiment).run()
