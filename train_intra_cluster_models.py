import sys
import argparse
import logging
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

    args = argparser.parse_args()

    return args


def main(num_clusters, num_words, vec_dim):
    df_master = pd.read_csv(f'datasets/GloVe/df_{num_clusters}_clusters_{num_words}.csv')
    df_master = df_master[:int(len(df_master)*0.9)]
    """
    COlumns of df_master
    Index(['word', 'vector', 'cluster_id', 'centroid_vector', 'nearest_0_id',
       'nearest_0_vec', 'nearest_1_id', 'nearest_1_vec', 'nearest_2_id',
       'nearest_2_vec', 'nearest_3_id', 'nearest_3_vec', 'nearest_4_id',
       'nearest_4_vec'],
      dtype='object')

    """
    
    for model_hash_index in range(num_clusters):
        x_train = []
        y_train = []
        
        df_for_given_cluster = df_master[df_master['cluster_id']==model_hash_index]
        skipped_training_data_rows = 0
        
        for idx in range(len(df_for_given_cluster)):
            x_query_vector = [float(s) for s in df_for_given_cluster.iloc[idx, 1].strip('][').split(', ')]
            x_centroid_vector = [float(s.strip()) for s in re.findall(r"[-+]?\d*\.\d+", df_for_given_cluster.iloc[idx, 3])]
            
            if len(x_query_vector) != 100:
                print(f"Skipped row! Cluster index {model_hash_index} x_centroid_vector is wrong len {len(x_query_vector)} should be 100")
                skipped_training_data_rows += 1
                continue           
                
            x_query_vector.extend(x_centroid_vector)
            if len(x_query_vector) != 200:
                print(f"Skipped row! Cluster index {model_hash_index} x_query_vector is wrong len {len(x_query_vector)} should be 200")
                skipped_training_data_rows += 1
                continue

            x_train.append(x_query_vector)
            y = int(df_for_given_cluster.iloc[idx, 4])
            y_train.append(y)

        print(f"Total rows skipped: {skipped_training_data_rows}")
        x_train = np.array(x_train)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        
        num_classes = len(set(y_train))
        y_train = tf.keras.utils.to_categorical(
            y_train#, num_classes=num_classes
        )

        model = Sequential([
          layers.Conv1D(128, 1, activation='relu', input_shape=(2*vec_dim,1),padding='same'),
          layers.MaxPooling1D(padding='same'),
          layers.Conv1D(128, 1, padding='same', activation='relu'),
          layers.Conv1D(256, 1, padding='same', activation='relu'),
          layers.Flatten(),
          layers.Dense(len(y_train[0]), activation="softmax")
        ])
        model.compile(
            loss="categorical_crossentropy", optimizer=SGD(), metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )

        
        # in model name, nl<Number> --> number of (nearest neighbor) labels, m --> number of members in the cluster
        mcp_save_callback = ModelCheckpoint(
     f"models/intra_cluster_models/{num_clusters}_clusters_{num_words}_num_words/model_2_{num_clusters}_clusters_{num_words}_num_words_modelhash_{model_hash_index}_nl{num_classes}_m{len(df_for_given_cluster)}",
            save_best_only=True,
            monitor="categorical_accuracy",
            mode="max",
        )
        
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=2, restore_best_weights=True)

        model.fit(x_train, y_train, steps_per_epoch=1024, epochs=500, callbacks=[mcp_save_callback, es_callback])
    

if __name__ == '__main__':
    initialize_logging()
    args = get_parse_args()
    main(args.num_clusters, args.num_words, args.vec_dim)