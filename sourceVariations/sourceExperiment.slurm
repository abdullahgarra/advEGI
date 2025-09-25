#!/bin/bash
#SBATCH --job-name=GML
#SBATCH --partition=cpu-killable
#SBATCH --nodelist=rack-ai-01
#SBATCH --cpus-per-task=20

#SBATCH --output=sourceExperiment.out
#SBATCH --error=sourceExperiment.err




echo "Totally random graph! 500 nodes..."
python3 sourceExperiment.py --file-path=data/random_graph_500.edgelist --label-path=data/random_labels.txt --n-dgi-epochs=100  --n-hidden=32 --self-loop --gpu=0 --n-layers=1 --dgi-lr=0.01 --model-id=2 --model-type=2
python3 sourceExperiment.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports.txt --data-id=europe --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1
python3 sourceExperiment.py --file-path=data/usa-airports.edgelist --label-path=data/labels-usa-airports.txt --data-id=usa --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1
python3 sourceExperiment.py --file-path=data/brazil-airports.edgelist --label-path=data/labels-brazil-airports.txt --data-id=brazil --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1
echo "================================================================"

echo "Running EU original..."
python3 sourceExperiment.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports.txt --n-dgi-epochs=100  --n-hidden=32 --self-loop --gpu=0 --n-layers=1 --dgi-lr=0.01 --model-id=2 --model-type=2
python3 sourceExperiment.py --file-path=data/usa-airports.edgelist --label-path=data/labels-usa-airports.txt --data-id=usa --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1
python3 sourceExperiment.py --file-path=data/brazil-airports.edgelist --label-path=data/labels-brazil-airports.txt --data-id=brazil --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1    

# echo "Evaluating original EU on degree-based labels"
python3 sourceExperiment.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports-degree-based.txt --data-id=europe --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1


# echo "Running EU with only 10 epoch ..."
# python3 runExperiment.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports.txt --n-dgi-epochs=10  --n-hidden=32 --self-loop --gpu=0 --n-layers=1 --dgi-lr=0.01 --model-id=2 --model-type=2
# python3 runExperiment.py --file-path=data/usa-airports.edgelist --label-path=data/labels-usa-airports.txt --data-id=usa --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1
# python3 runExperiment.py --file-path=data/brazil-airports.edgelist --label-path=data/labels-brazil-airports.txt --data-id=brazil --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1       


#=====================================================================================================================
# echo "Trying model type = 1"
# python3 2passTrain.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports.txt --n-dgi-epochs=100  --n-hidden=32 --self-loop --gpu=0 --n-layers=1 --dgi-lr=0.01 --model-id=0 --model-type=0
# python3 2passTrain.py --file-path=data/usa-airports.edgelist --label-path=data/labels-usa-airports.txt --data-id=usa --data-src=europe --model-type=0 --model-id=0 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1
# python3 2passTrain.py --file-path=data/brazil-airports.edgelist --label-path=data/labels-brazil-airports.txt --data-id=brazil --data-src=europe --model-type=0 --model-id=0 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1    
