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
#sh TrainEval.sh "$EXPERIMENT_NAME" config_test-Atomiser_Atos.yaml regular

MODEL_NAME=config_test-Atomiser_Atos.yaml

sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_1
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_2
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_3
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_4
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_5
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_6
