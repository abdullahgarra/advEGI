#!/bin/bash
#SBATCH --job-name=GML
#SBATCH --partition=cpu-killable
#SBATCH --nodelist=rack-ai-01
#SBATCH --cpus-per-task=20

#SBATCH --output=compute_bound_filepath.out
#SBATCH --error=compute_bound_filepath.err

echo "DELTA: EU to USA"
python3 compute_bound_filepath.py --file-path=linearProbes/data/europe-airports.edgelist --label-path=linearProbes/data/usa-airports.edgelist

echo "========================================================"
echo "DELTA: same degree to USA"
python3 compute_bound_filepath.py --file-path=linearProbes/data/same-degree.random.edgelist--label-path=linearProbes/data/usa-airports.edgelist

echo "========================================================"
echo "DELTA: EU to Brazil"
python3 compute_bound_filepath.py --file-path=linearProbes/data/airports.edgelist --label-path=linearProbes/data/brazil-airports.edgelist

echo "========================================================"
echo "DELTA: same degree to Brazil"
python3 compute_bound_filepath.py --file-path=linearProbes/data/same-degree.random.edgelist --label-path=linearProbes/data/brazil-airports.edgelist

echo "========================================================"
echo "DELTA: EU to same degree"
python3 compute_bound_filepath.py --file-path=linearProbes/data/europe-airports.edgelist --label-path=linearProbes/data/same-degree.random.edgelist

echo "========================================================"
echo "DELTA: same degree to EU"
python3 compute_bound_filepath.py --file-path=linearProbes/data/same-degree.random.edgelist --label-path=linearProbes/data/europe-airports.edgelist


echo "Delta: Random -> EU"
python3 compute_bound_filepath.py --file-path=linearProbes/data/random_graph_500.edgelist --label-path=linearProbes/data/europe-airports.edgelist

echo "Delta: random -> usa"
python3 compute_bound_filepath.py --file-path=linearProbes/data/random_graph_500.edgelist --label-path=linearProbes/data/usa-airports.edgelist

echo "Delta: random -> brazil"
python3 compute_bound_filepath.py --file-path=linearProbes/data/random_graph_500.edgelist --label-path=linearProbes/data/brazil-airports.edgelist
