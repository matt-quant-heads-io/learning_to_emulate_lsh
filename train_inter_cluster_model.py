import sys
import logging
import argparse


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
    df = pd.read_csv(f'datasets/GloVe/df_{num_clusters}_clusters_{num_words}.csv')
    """
    Index(['word', 'vector', 'cluster_id', 'centroid_vector', 'nearest_0_id',
           'nearest_0_vec', 'nearest_1_id', 'nearest_1_vec', 'nearest_2_id',
           'nearest_2_vec', 'nearest_3_id', 'nearest_3_vec', 'nearest_4_id',
           'nearest_4_vec'],
          dtype='object')

    """

    x_train = []
    y_train = []
    data_len = int(len(df)*0.9)
    for idx in range(data_len):
        x = [float(s) for s in df.iloc[idx, 1].strip('][').split(', ')]
        y = int(df.iloc[idx, 2])
        x_train.append(x)
        y_train.append(y)


    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    y_train = tf.keras.utils.to_categorical(
        y_train, num_classes=num_clusters
    )

    num_classes = num_clusters
    embedding_dim = 100


    model = Sequential([
      layers.Conv1D(128, 1, activation='relu', input_shape=(vec_dim,1),padding='same'),
      layers.MaxPooling1D(padding='same'),
      layers.Conv1D(128, 1, padding='same', activation='relu'),
      layers.Conv1D(256, 1, padding='same', activation='relu'),
      layers.Flatten(),
      layers.Dense(num_clusters, activation="softmax")
    ])
    model.compile(
        loss="categorical_crossentropy", optimizer=SGD(), metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    mcp_save_callback = ModelCheckpoint(
        f"models/inter_cluster_models/{num_clusters}_clusters_{num_words}_num_words/model_1_{num_clusters}_clusters_{num_words}_num_words",
        save_best_only=True,
        monitor="categorical_accuracy",
        mode="max",
    )


    es_callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=7, restore_best_weights=True)

    model.fit(x_train, y_train, steps_per_epoch=1024, epochs=500, callbacks=[mcp_save_callback, es_callback])

if __name__ == '__main__':
    initialize_logging()
    args = get_parse_args()
    main(args.num_clusters, args.num_words, args.vec_dim)





