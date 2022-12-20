#!/bin/sh
# echo "Creating datasets dirs"

CURR_DIR="$(pwd)"

# for subdir in GloVe
# do
#   mkdir -p $CURR_DIR/datasets/$subdir
# done


# echo "\n\nCreating models dirs"
# for subdir in  vision language 
# do
#   mkdir -p $CURR_DIR/models/$subdir
# done


# TEMP_MODEL_ZIP=flower_model_bit_0.96875.zip

# wget -q https://git.io/JuMq0 -O $TEMP_MODEL_ZIP
# unzip -qq $TEMP_MODEL_ZIP

# rm -rf models/vision/*
# mv flower_model_bit_0.96875 models/vision
# rm $TEMP_MODEL_ZIP

# curl -L http://www-nlp.stanford.edu/data/glove.6B.zip -o $CURR_DIR/glove.6B.zip
unzip $CURR_DIR/glove.6B.zip -d $CURR_DIR/datasets/GloVe
rm data/glove.6B.zip

echo "Downloading datasets"
# python3 bin/download_datasets.py








