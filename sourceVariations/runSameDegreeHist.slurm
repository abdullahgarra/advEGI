#!/bin/bash
#SBATCH --job-name=SameDegree
#SBATCH --partition=cpu-killable
#SBATCH --nodelist=rack-ai-01
#SBATCH --cpus-per-task=20

#SBATCH --output=runSameDegreeHist.out
#SBATCH --error=runSameDegreeHist.err


echo "Same Degree Histogram..."
python3 runSameDegreeHist.py --file-path=data/same-degree.random.edgelist --label-path=data/labels-europe-airports.txt --n-dgi-epochs=100  --n-hidden=32 --self-loop --gpu=0 --n-layers=1 --dgi-lr=0.01 --model-id=2 --model-type=2

python3 runSameDegreeHist.py --file-path=data/brazil-airports.edgelist --label-path=data/labels-brazil-airports.txt --n-dgi-epochs=0  --n-hidden=32 --self-loop --gpu=0 --n-layers=1 --dgi-lr=0.01 --model-id=2 --model-type=2 --n-classifier-epochs=100
python3 runSameDegreeHist.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports.txt --data-id=europe --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1
python3 runSameDegreeHist.py --file-path=data/usa-airports.edgelist --label-path=data/labels-usa-airports.txt --data-id=brazil --data-src=europe --model-type=2 --model-id=2 --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1
echo "================================================================"
