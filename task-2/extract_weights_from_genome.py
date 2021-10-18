import pickle, numpy
from NEAT_evoman import EvomanNEAT

if __name__ == '__main__':
    genome_path = 'NEAT-final-results/v1/NEAT-v1-15_10_2021_09_01_39/round-1/highscores/gen25_genome639(4.3_fitness).pk1'

    with open(genome_path, 'rb') as output:
        genome = pickle.load(output)
    weights = EvomanNEAT.weights_from_genome(genome, genome_path.__contains__('NEAT-v2'))
    numpy.savetxt(f'{genome_path.split(".pk1")[0]}.txt', weights)