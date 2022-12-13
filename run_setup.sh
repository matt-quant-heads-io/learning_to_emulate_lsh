#!/bin/sh
echo "Creating datasets dirs"

CURR_DIR="$(pwd)"

for subdir in GloVe
do
  mkdir -p $CURR_DIR/datasets/$subdir
done


echo "\n\nCreating models dirs"
for subdir in  vision language 
do
  mkdir -p $CURR_DIR/models/$subdir
done

curl -L http://www-nlp.stanford.edu/data/glove.6B.zip -o $CURR_DIR/datasets/glove.6B.zip
unzip $CURR_DIR/datasets/glove.6B.zip -d $CURR_DIR/datasets/GloVe
rm $CURR_DIR/datasets/glove.6B.zip

# echo "Downloading datasets"
# python3 bin/download_datasets.py








