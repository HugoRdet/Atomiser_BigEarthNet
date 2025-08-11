#!/bin/bash
source /etc/profile.d/lmod.sh
module load conda



## === Then load the module and activate your env ===
conda activate venv


#sh TrainEval.sh test_Atos_lancement_scale config_test-ScaleMAE.yaml regular



MODEL_NAME=config_test-Atomiser_Atos.yaml


sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_8


MODEL_NAME=config_test-ViT_XS.yaml


sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_8


MODEL_NAME=config_test-ResNet50.yaml


sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_8

MODEL_NAME=config_test-Perceiver.yaml


sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_8

MODEL_NAME=config_test-ScaleMAE.yaml

sh Eval_modalities.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_8