#!/bin/bash
#SBATCH --job-name=runPGDattack
#SBATCH --partition=cpu-killable
#SBATCH --nodelist=rack-ai-01
#SBATCH --cpus-per-task=20
#SBATCH --output=runPGDattack.out
#SBATCH --error=runPGDattack.err



echo "BASELINE"
python3 runPGDattack.py \
  --model-type 2 --model-id 2 \
  --file-path data/europe-airports.edgelist \
  --label-path data/labels-europe-airports.txt \
  --n-dgi-epochs 100 --pgd-steps 0 \
  --eval-targets usa,brazil --eval-head-epochs 300 \
  --seed 1

echo "PGD mild"
python3 runPGDattack.py \
  --model-type 2 --model-id 2 \
  --file-path data/europe-airports.edgelist \
  --label-path data/labels-europe-airports.txt \
  --n-dgi-epochs 10 --pgd-steps 1 --pgd-add-k 8 --pgd-del-k 8 --pgd-rand-subset 100 --pgd-degree-cap 4 \
  --eval-targets usa,brazil --eval-head-epochs 300 \
  --seed 1


echo "PGD aggresive"
python3 runPGDattack.py \
  --model-type 2 --model-id 2 \
  --file-path data/europe-airports.edgelist \
  --label-path data/labels-europe-airports.txt \
  --n-dgi-epochs 10 \
  --pgd-steps 3 \
  --pgd-add-k 16 --pgd-del-k 16 \
  --pgd-rand-subset 300 \
  --pgd-degree-cap 10 \
  --features-recompute-per-epoch 1 \
  --eval-targets usa,brazil --eval-head-epochs 300 \
  --seed 1