#!/bin/bash
source /etc/profile.d/lmod.sh
module load conda



## === Then load the module and activate your env ===
conda activate venv


#sh TrainEval.sh test_Atos_lancement_scale config_test-ScaleMAE.yaml regular



MODEL_NAME=config_test-ScaleMAE.yaml

#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_1
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_2
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_3
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_4
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_5
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" add_6
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" test_1
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" test_2
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" test_3
#sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" train_1
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" train_2
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" train_3
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" val_1
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" val_2
sh TrainEval.sh "$EXPERIMENT_NAME" "$MODEL_NAME" val_3