import sys
import os
import logging
import argparse
import random
import re
import json


from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
import keras
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
from numpy import unique
import pandas as pd
from keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import requests
import tempfile
import time
import scann




def initialize_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s> %(filename)s::%(funcName)s::%(lineno)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    

def get_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--vec_dim", type=int, default=100)
    argparser.add_argument("--num_runs", type=int, default=100)
       
    args = argparser.parse_args()

    return args


def run_random_experiment(test_data_df, num_words, num_clusters, num_runs, final_results):
    model_1 = keras.models.load_model(f"models/inter_cluster_models/{num_clusters}_clusters_{num_words}_num_words/model_1_{num_clusters}_clusters_{num_words}_num_words") 
    results = {"run": [], "actual_distance": [], "predicted_distance": []}
    json_clusters = json.load(open(f"/home/jupyter-msiper/algorithmic_ml_sandbox/datasets/GloVe/100D_{num_words}-words_{num_clusters}-clusters.json"))

    
    test_query_vector = None
    test_centroid_vector = None
    for run in range(num_runs):
        test_idx = random.randint(0, num_words//2)
        other_test_idx = random.randint(num_words//2, num_words)
        test_query_vector = [float(s) for s in test_data_df.iloc[test_idx, 1].strip('][').split(', ')]
        test_centroid_vector = [float(s.strip()) for s in re.findall(r"[-+]?\d*\.\d+", test_data_df.iloc[test_idx, 3])]

        test_nearest_neighbor_vec = [float(s.strip()) for s in re.findall(r"[-+]?\d*\.\d+", test_data_df.iloc[test_idx, 5])]
        actual_distance = sum([np.abs(test_nearest_neighbor_vec[i] - test_query_vector[i]) for i in range(100)])
        
        
        model_1_test_vector = np.array([test_query_vector])
        model_1_prediction = np.argmax(model_1.predict(model_1_test_vector))
        
        test_query_vector.extend(test_centroid_vector)
        model_2_test_vector = np.array([test_query_vector])
        
        
        
        model_2_vec = [float(s) for s in test_data_df.iloc[other_test_idx, 1].strip('][').split(', ')]
        predicted_distance = sum([float(np.abs(test_nearest_neighbor_vec[i] - model_2_vec[i])) for i in range(100)])
        absolute_difference = np.abs(actual_distance - predicted_distance)
        
        results["run"].append(run)
        results["actual_distance"].append(actual_distance)
        results["predicted_distance"].append(predicted_distance)
        
    final_results["random"].append(sum(results["predicted_distance"])/num_runs)

    return final_results
    
    
    
def run_ml_experiment(test_data_df, num_words, num_clusters, num_runs, final_results):
    model_1 = keras.models.load_model(f"models/inter_cluster_models/{num_clusters}_clusters_{num_words}_num_words/model_1_{num_clusters}_clusters_{num_words}_num_words") 
    results = {"run": [], "actual_distance": [], "predicted_distance": []}
    json_clusters = json.load(open(f"/home/jupyter-msiper/algorithmic_ml_sandbox/datasets/GloVe/100D_{num_words}-words_{num_clusters}-clusters.json"))
    
    test_query_vector = None
    test_centroid_vector = None
    for run in range(num_runs):
        test_idx = random.randint(0, num_words)
        test_query_vector = [float(s) for s in test_data_df.iloc[test_idx, 1].strip('][').split(', ')]
        test_centroid_vector = [float(s.strip()) for s in re.findall(r"[-+]?\d*\.\d+", test_data_df.iloc[test_idx, 3])]
#         test_centroid_vector = test_centroid_vector / np.linalg.norm(test_centroid_vector, axis=0)
        print(f"test_centroid_vector: {test_centroid_vector}")
        test_nearest_neighbor_vec = [float(s.strip()) for s in re.findall(r"[-+]?\d*\.\d+", test_data_df.iloc[test_idx, 5])]

        actual_distance = sum([np.abs(test_nearest_neighbor_vec[i] - test_query_vector[i]) for i in range(100)])
        
        
        model_1_test_vector = np.array([test_query_vector])
        model_1_prediction = np.argmax(model_1.predict(model_1_test_vector))
        
        test_query_vector.extend(test_centroid_vector)
        model_2_test_vector = np.array([test_query_vector])
        
        path_to_model_2 = None
        model_2_dirs = os.listdir(f"/home/jupyter-msiper/algorithmic_ml_sandbox/models/intra_cluster_models/{num_clusters}_clusters_{num_words}_num_words")

        
        for model_2_dir in model_2_dirs:
            if model_2_dir.startswith(f"model_2_{num_clusters}_clusters_{num_words}_num_words_modelhash_{model_1_prediction}"):
                path_to_model_2 = f"/home/jupyter-msiper/algorithmic_ml_sandbox/models/intra_cluster_models/{num_clusters}_clusters_{num_words}_num_words/{model_2_dir}"
                
        model_2 = keras.models.load_model(path_to_model_2) 
        model_2_prediction = np.argmax(model_2.predict(model_2_test_vector))

        model_2_df = test_data_df[test_data_df['cluster_id']==model_1_prediction]
        
        tries = 0
        
        if model_2_prediction >= len(json_clusters[model_1_prediction]):
            continue
            
        model_2_vec = json_clusters[model_1_prediction][model_2_prediction]
        model_2_df = test_data_df[test_data_df['word']==model_2_vec]
        
        model_2_vec = [float(s) for s in model_2_df.iloc[0, 1].strip('][').split(', ')]
        predicted_distance = sum([float(np.abs(test_nearest_neighbor_vec[i] - model_2_vec[i])) for i in range(100)])
        absolute_difference = np.abs(actual_distance - predicted_distance)
        
        results["run"].append(run)
        results["actual_distance"].append(actual_distance)
        results["predicted_distance"].append(predicted_distance)
        
    final_results["runs"].append(num_runs)
    final_results["ml"].append(sum(results["predicted_distance"])/num_runs)

    return final_results
        
        
def run_intra_cluster_experiment(test_data_df, num_words, num_runs):
    pass


def run_inter_cluster_experiment(test_data_df, num_words, num_runs):
    pass


def run_scann_experiment(test_data_df, num_words, num_clusters, num_runs, final_results):
    loc = os.path.join("/home/jupyter-msiper/algorithmic_ml_sandbox/", "glove.hdf5")   
    glove_h5py = h5py.File(loc, "r")
    
    dataset = glove_h5py['train']
    queries = glove_h5py['test']
    
#     dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]

    searcher = scann.scann_ops_pybind.builder(dataset, num_clusters, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=num_words).score_ah(
        2, anisotropic_quantization_threshold=0.01).reorder(100).build()
    
    results = {"run": [], "predicted_distance": []}

    for run in range(num_runs):
        results["run"].append(run)
        test_idx = random.randint(0, num_words)
        
        neighbors, distances = searcher.search(dataset[test_idx], final_num_neighbors=1)
        results["predicted_distance"].append(distances[0])
    final_results["scann"].append(sum(results["predicted_distance"])/len(results["predicted_distance"]))
    
    return final_results
    
    
def run_experiment(num_clusters, num_words, vec_dim, num_runs):
    final_results = {"runs": [], "scann":[], "random":[], "ml": []}
    df = pd.read_csv(f'datasets/GloVe/df_{num_clusters}_clusters_{num_words}.csv')
    for exp_name, func in EXPERIMENT_FUNC_DICT.items():
        final_results = func(df, num_words, num_clusters, num_runs, final_results)
        
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(f"experiment_{num_words}_num_words_{num_clusters}_num_clusters_{1000}_num_runs.csv")
    

def main(vec_dim, num_runs):
    num_clusters_words = [(10, 10000), (100, 10000), (10, 20000), (100, 20000), (10, 50000), (100, 50000)]
    
    for num_clusters, num_words in num_clusters_words:
        run_experiment(num_clusters, num_words, vec_dim, num_runs)
        

if __name__ == '__main__':
    EXPERIMENT_FUNC_DICT = {
        "run_ml_experiment": run_ml_experiment,
        "run_random_experiment": run_random_experiment,
        "scann": run_scann_experiment,
    }
    
    initialize_logging()
    args = get_parse_args()
    
    main(args.vec_dim, args.num_runs)