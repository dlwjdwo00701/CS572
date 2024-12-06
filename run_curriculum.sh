#!/bin/bash

# Shell script to run the DreamerV3 training with specified arguments.
 # for subssequeeent training i will use this as the starting pretrained
python -m embodied.agents.dreamerv3.train \
  --configs humanoid_benchmark \
  --run.wandb True \
  --run.num_envs 1 \
  --run.script curriculum_learning \
  --run.eval_every 5000 \
  --run.wandb_entity adeeb-islam8 \
  --method dreamer \
  --logdir logs \
  --task humanoid_h1hand-stand-v0 \
  --seed 0 \
  --batch_size 4 \
  --replay_size 1e6 \
  --run.steps 1e6 \
  --run.from_checkpoint "" 
