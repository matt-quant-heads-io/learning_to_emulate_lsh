# Learning to Emulate Locality Sensitive Hashing for Nearest Neighbor Search: An Empirical Comparison

In this work we emulate nearest neighbor search with a multihead neural network that predicts inter-cluster then intra-cluster labels generated from a fitted K-means model on the GloVe dataset. Paper [here](EmulatingNearestNeighborSearch.pdf).

## Setup 
1) Create conda environemnt from terminal

`conda create -n algorithmic_ml_sandbox python=3.7`

2) Activate the environment

`conda activate algorithmic_ml_sandbox`

3) Install all of the dependencies via running the following command at the project root

`pip install -e .`

4) Create all of the directories and download the datasets

`sh run_setup.sh`

## Generate a datasets

1) At project root run 

`python3 word_clustering.py --num_clusters 10 --vector_dim 100 --num_words 20000 --glove_path /path/to/your/datasets/folder`

## Train ML System on a given dataset 

1) Train layer 1 model 

`python3 train_inter_cluster_model.py --num_clusters 10 --num_words 20000 --vec_dim 100`

2) Train layer 2 models

`python3 train_intra_cluster_model.py --num_clusters 10 --num_words 20000 --vec_dim 100`

## Run the experiments 

1) Run the experiments trained model against ScaNN system using the generated datasets

`python3 run_experiments.py --num_runs 1000 --vec_dim 100`


