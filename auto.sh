#!/bin/bash

set -e

gpu=$1


start=$(date "+%s")

#CUDA_VISIBLE_DEVICES=$gpu python create_training_data.py --config config/training_config.yaml

#CUDA_VISIBLE_DEVICES=$gpu python train_aligner.py --config config/training_config.yaml

CUDA_VISIBLE_DEVICES=$gpu python extract_durations.py --config config/training_config.yaml

#CUDA_VISIBLE_DEVICES=$gpu python train_tts.py --config config/training_config.yaml

end=$(date "+%s")
time=$((end-start))
echo "time used:$time seconds"
