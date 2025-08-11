#!/bin/bash
source /etc/profile.d/lmod.sh
module load conda

# Generate a random experiment name if none is provided
if [ -z "$1" ]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RANDOM_SUFFIX=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1)
  EXPERIMENT_NAME="xp_${TIMESTAMP}_${RANDOM_SUFFIX}"
  echo "No experiment name provided. Using generated name: $EXPERIMENT_NAME"
else
  EXPERIMENT_NAME=$1
fi

## === Then load the module and activate your env ===
conda activate venv

# Call training script with experiment name used in the arguments
#sh TrainEval.sh "$EXPERIMENT_NAME" config_test-Perceiver.yaml regular

MODEL_NAME=config_test-Perceiver.yaml

sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r1
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r2
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r3
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r4
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r5
sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" Big_r6