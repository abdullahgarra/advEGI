#!/bin/bash
#SBATCH --job-name=linearPredictorWithLogs
#SBATCH --partition=cpu-killable
#SBATCH --cpus-per-task=20
#SBATCH --output=linearPredictorWithLogs.out
#SBATCH --error=linearPredictorWithLogs.err


echo "normal"
python3 linearPredictorWithLogs.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports.txt \
  --data-id=europe --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=100 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=degree --probe-degree \
  --out-dir=results --run-tag=normal


python3 linearPredictorWithLogs.py --file-path=data/usa-airports.edgelist --label-path=data/labels-usa-airports.txt \
  --data-id=usa --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=degree --probe-degree --residualize \
  --out-dir=results --run-tag=normal


python3 linearPredictorWithLogs.py --file-path=data/brazil-airports.edgelist --label-path=data/labels-brazil-airports.txt \
  --data-id=brazil --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=degree --probe-degree --residualize \
  --out-dir=results --run-tag=normal

echo " ======================================================================"

echo "residualized"
python3 linearPredictorWithLogs.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports.txt \
  --data-id=europe --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=degree --probe-degree --residualize \
  --out-dir=results --run-tag=residualized


python3 linearPredictorWithLogs.py --file-path=data/usa-airports.edgelist --label-path=data/labels-usa-airports.txt \
  --data-id=usa --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=degree --probe-degree --residualize \
  --out-dir=results --run-tag=residualized


python3 linearPredictorWithLogs.py --file-path=data/brazil-airports.edgelist --label-path=data/labels-brazil-airports.txt \
  --data-id=brazil --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=degree --probe-degree --residualize \
  --out-dir=results --run-tag=residualized

echo " ======================================================================"

echo "bypass encoder + keep degree"
  python3 linearPredictorWithLogs.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports.txt \
  --data-id=europe --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=degree --bypass-encoder \
  --out-dir=results --run-tag=bypass_encoder

python3 linearPredictorWithLogs.py --file-path=data/usa-airports.edgelist --label-path=data/labels-usa-airports.txt \
  --data-id=usa --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=degree --bypass-encoder \
  --out-dir=results --run-tag=bypass_encoder

python3 linearPredictorWithLogs.py --file-path=data/brazil-airports.edgelist --label-path=data/labels-brazil-airports.txt \
  --data-id=brazil --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=degree  --bypass-encoder \
  --out-dir=results --run-tag=bypass_encoder
echo " ======================================================================"

echo "bypass encoder + make constants"
  python3 linearPredictorWithLogs.py --file-path=data/europe-airports.edgelist --label-path=data/labels-europe-airports.txt \
  --data-id=europe --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=constant --bypass-encoder \
  --out-dir=results --run-tag=bypass_encoder_constant

python3 linearPredictorWithLogs.py --file-path=data/usa-airports.edgelist --label-path=data/labels-usa-airports.txt \
  --data-id=usa --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=constant --bypass-encoder \
  --out-dir=results --run-tag=bypass_encoder_constant

python3 linearPredictorWithLogs.py --file-path=data/brazil-airports.edgelist --label-path=data/labels-brazil-airports.txt \
  --data-id=brazil --data-src=europe --model-type=2 --model-id=2 \
  --n-dgi-epochs=0 --n-classifier-epochs=100 --n-hidden=32 --self-loop --gpu=-1 \
  --feature-mode=constant  --bypass-encoder \
  --out-dir=results --run-tag=bypass_encoder_constant
