#!/bin/bash
for experiment in "$@"
do
  python NEAT_experiment.py "$experiment" &
done