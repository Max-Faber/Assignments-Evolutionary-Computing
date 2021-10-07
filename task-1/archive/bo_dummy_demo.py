################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os

sys.path.insert(0, 'evoman')

import numpy as np
from time import time, strftime, localtime
from environment import Environment
from demo_controller import player_controller

headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
n_hidden_neurons = 10
# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[3],  # 3 enemies/games
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  logs="off",
                  speed="fastest")

number_of_rounds = 10

# todo: the optimal vars (hyperparams) need to found somehow (literary search?)
num_gen = 5
num_pop = 500
num_vars = ((env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5)
tournament_size = int(num_pop / 2)
crossover_prob = 0.9
mutation_prob = 0.9
num_mutations = int(num_vars * 0.01)
bounds_min = -1
bounds_max = 1


# generates a random individual, based on preconfigured gene boundaries and # of genes
def random_individual():
    return np.random.uniform(bounds_min, bounds_max, num_vars)


# initializes a random set of individuals (population), within the gene's boundaries
def random_population():
    pop = []
    for _ in range(num_pop):
        pop.append(random_individual())
    return pop


# Tournament selection
def select_parent(population, scores):
    # select a random parent in the population
    best_parent = np.random.randint(0, num_pop - 1)
    # select n random parents in the population
    samples = []
    for _ in range(tournament_size):
        samples.append(population[np.random.randint(0, num_pop - 1)])
    # check for each selected parent, whether it is better than the current best parent from the random selection
    for s in range(len(samples)):
        current_evaluate = scores[s]
        best_parent_evaluate = scores[best_parent]
        # a higher evaluation is better
        if current_evaluate > best_parent_evaluate:
            best_parent = s
    # return the best parent from a random subset of the population
    return population[best_parent]


# Does a crossover between two parents to generate two new children
def do_crossover(parent1, parent2):
    # Duplicate the parents to the children, respectively
    child1 = parent1.copy()
    child2 = parent2.copy()
    # Check if we do want to do a crossover, probability based
    if np.random.uniform(0, 1) > (1 - crossover_prob):
        # Do a uniform crossover, i.e. randomly selecting from one of the parents.
        for child in [child1, child2]:
            for gene_idx, val in enumerate(child):
                child[gene_idx] = parent1[gene_idx] if np.random.uniform(0, 1) > 0.5 else parent2[gene_idx]
    return child1, child2


# Mutates given child by swapping one of the genes (randomly)
def mutate(child):
    # Check if we want to mutate (probability based)
    if np.random.uniform(0, 1) > (1 - mutation_prob):
        # Choose one of the genes to swap randomly
        mutated_genes = []
        for _ in range(num_mutations):
            random_gene = np.random.randint(0, len(child) - 1)
            while mutated_genes.__contains__(random_gene):
                random_gene = np.random.randint(0, len(child) - 1)
            mutated_genes.append(random_gene)
            # Swap the gene with a new uniform one (within the min/max gene's bounds)
            child[random_gene] = np.random.uniform(bounds_min, bounds_max)
    # Return the (possibly) mutated child
    return child


# evaluate performance of player, higher is better (player fitness - enemy fitness)
def evaluate(x):
    _, pl, el, _ = env.play(pcont=x)
    indv_gain = pl - el
    return indv_gain


def evolve(gen_old, gen_performance_old):
    # Initalize the collections of the new generation
    new_generation = []
    new_generation_fitness = []
    # Create an entire new population
    for gen in range(0, num_pop - 1, 2):
        # Select two parents using tournament selection
        parent1, parent2 = select_parent(gen_old, gen_performance_old), select_parent(gen_old, gen_performance_old)
        # Create two childs (with crossover) between the two parents (to keep the population size)
        for child in do_crossover(parent1, parent2):
            # Mutate the child
            child = mutate(child)
            # Append the child and its fitness to the new population
            new_generation.append(child)
            new_generation_fitness.append(evaluate(child))
    return np.asarray(new_generation), np.asarray(new_generation_fitness)


# usage: show_best(np.loadtxt('somepathtoyourbestindividual'))
def show_best(best):
    env.update_parameter('speed', 'normal')
    evaluate(best)


# global vars
train_name = strftime("%d_%m_%Y_%H_%M_%S", localtime(time()))
pop = []
fitness = []
best_individual = []

# for every generation, evaluate the population
for i in range(num_gen):
    gen_name = "gen_{}".format(i)
    if len(pop) == 0:
        print "First population, random initialization"
        pop = random_population()
        for p in pop:
            fitness.append(evaluate(p))
    print "Training gen {}. Pop. size: {}.".format(i, num_pop)
    pop, fitness = evolve(pop, fitness)
    avg_fitness = fitness.sum() / len(fitness)
    max_fitness = fitness.max()
    best_individual = fitness.tolist().index(max_fitness)

    folder = experiment_name + '/' + train_name + '/' + gen_name
    file_name = folder + '/best_fitness_{}.txt'.format(max_fitness)
    os.makedirs(folder)
    np.savetxt(file_name, pop[best_individual])
    print "Current best individual's fitness: {}. Avg. population's fitness: {}.".format(max_fitness, avg_fitness)