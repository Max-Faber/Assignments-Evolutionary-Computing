import sys

sys.path.insert(0, 'evoman')
from time import time, strftime, localtime
from NEAT_evoman import EvomanNEAT

experiments = [
    {
        "name": "NEAT-v1",
        "config-file": "NEAT-configs/config-feedforward-1.txt",
        "3-enemies": [1, 2, 3],
        "number-of-generations": 10
    },
    {
        "name": "NEAT-v2",
        "config-file": "NEAT-configs/config-feedforward-2.txt",
        "3-enemies": [1, 2, 3],
        "number-of-generations": 10
    }
]

if __name__ == '__main__':
    dt = strftime("%d_%m_%Y_%H_%M_%S", localtime(time()))
    for experiment in experiments:
        name = experiment["name"]
        config = experiment["config-file"]
        print("Running experiment: {}, config file used: {}".format(name, config))
        for enemy in experiment["3-enemies"]:
            env_dir = "NEAT-results/" + name + "/" + dt + "/enemy-" + str(enemy)
            num_gens = experiment["number-of-generations"]

            n = EvomanNEAT(neat_config=experiment["config-file"],
                           number_of_gens=int(num_gens),
                           experiment_env=env_dir,
                           enemy=enemy)

            winner = n.run()
