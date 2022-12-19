import sys
import os
import logging
import argparse
import random
import re


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




def initialize_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s> %(filename)s::%(funcName)s::%(lineno)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    

def get_parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_clusters", type=int, required=True)
    argparser.add_argument("--num_words", type=int, required=True)
    argparser.add_argument("--vec_dim", type=int, default=100)
    argparser.add_argument("--experiment", required=True, choices=["run_ml_experiment", "intra_cluster", "inter_cluster", "scann"])
    argparser.add_argument("--num_runs", type=int, default=100)
        

    args = argparser.parse_args()

    return args

    
def run_ml_experiment(test_data_df, num_words, num_clusters, num_runs):
    model_1 = keras.models.load_model(f"models/inter_cluster_models/{num_clusters}_clusters_{num_words}_num_words/model_1_{num_clusters}_clusters_{num_words}_num_words") 
    results = {"run": [], "actual_distance": [], "predicted_distance": []}
    final_results = {"total_runs": [], "avg_predicted_difference": []}
    
    # select the vector of the correct answer
    test_query_vector = None
    test_centroid_vector = None
    for run in range(num_runs):
        test_idx = random.randint(0, num_words)
        test_query_vector = [float(s) for s in test_data_df.iloc[test_idx, 1].strip('][').split(', ')]
        test_centroid_vector = [float(s.strip()) for s in re.findall(r"[-+]?\d*\.\d+", test_data_df.iloc[test_idx, 3])]
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
        print(f"model_2_prediction: {model_2_prediction}")
        model_2_df = test_data_df[test_data_df['cluster_id']==model_1_prediction]
        print(model_2_df.head())
        model_2_vec = [float(s) for s in model_2_df.iloc[model_2_prediction, 1].strip('][').split(', ')]
        predicted_distance = sum([np.abs(test_nearest_neighbor_vec[i] - model_2_vec[i]) for i in range(100)])
        absolute_difference = np.abs(actual_distance - predicted_distance)
        
        results["run"].append(run)
        results["actual_distance"].append(actual_distance)
        results["predicted_distance"].append(predicted_distance)
        
    final_results["total_runs"].append(num_runs)
    final_results["avg_predicted_difference"].append(sum(results["predicted_distance"])/num_runs)
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(f"ml_exp_{num_words}_num_words_{num_clusters}_num_clusters_{1000}_num_runs.csv")
        
        
def run_intra_cluster_experiment(test_data_df, num_words, num_runs):
    pass

def run_inter_cluster_experiment(test_data_df, num_words, num_runs):
    pass


def run_scann_experiment(test_data_df, num_words, num_runs):
    pass
    
    
def main(num_clusters, num_words, vec_dim, experimemt, num_runs):
    df = pd.read_csv(f'datasets/GloVe/df_{num_clusters}_clusters_{num_words}.csv')
    EXPERIMENT_FUNC_DICT[experimemt](df, num_words, num_clusters, num_runs)

if __name__ == '__main__':
    EXPERIMENT_FUNC_DICT = {
    "run_ml_experiment": run_ml_experiment,
    "intra_cluster": run_intra_cluster_experiment,
    "inter_cluster": run_inter_cluster_experiment,
    "scann": run_scann_experiment,
}
    
    initialize_logging()
    args = get_parse_args()
    
    main(args.num_clusters, args.num_words, args.vec_dim, args.experiment, args.num_runs)