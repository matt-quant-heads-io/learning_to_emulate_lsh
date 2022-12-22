#!/bin/sh

echo "Creating datasets dirs"

CURR_DIR="$(pwd)"

for subdir in GloVe
do
  mkdir -p $CURR_DIR/datasets/$subdir
done

curl -L http://www-nlp.stanford.edu/data/glove.6B.zip -o $CURR_DIR/datasets/glove.6B.zip
unzip $CURR_DIR/datasets/glove.6B.zip -d $CURR_DIR/datasets/GloVe
rm $CURR_DIR/datasets/glove.6B.zip

echo "Downloaded GloVe datasets"

echo "Creating models directories"

for subdir in 10_clusters_10000_num_words 100_clusters_10000_num_words 10_clusters_20000_num_words 100_clusters_20000_num_words 10_clusters_50000_num_words 100_clusters_50000_num_words
do
  mkdir -p $CURR_DIR/models/inter_cluster_models/$subdir
done

echo "Creating experiments directories"
for subdir in results
do
  mkdir -p $CURR_DIR/experiments/$subdir
done








