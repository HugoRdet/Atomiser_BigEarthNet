#!/bin/bash
source /etc/profile.d/lmod.sh
module load conda



## === Then load the module and activate your env ===
conda activate venv


sh TrainEval.sh test_Atos_lancement_scale config_test-ScaleMAE_low_lr.yaml regular



