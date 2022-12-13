import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

import tensorflow_datasets as tfds
import pandas as pd

import sys, os
sys.path.append('/Users/matt/algorithmic_ml_sandbox/')

print(sys.path)
from lib.lsh.lsh import BuildLSHTable

tfds.disable_progress_bar()

IMAGE_SIZE = 224
NUM_IMAGES = 1000


def get_train_test_dfs(path_to_train_test_csvs):
    df_train = pd.read_csv(f'{path_to_train_test_csvs}/train.csv', header=[0])

    labels = []
    rows_data = []
    for idx in range(len(df_train)):
        row_data = np.array(df_train.iloc[idx, :df_train.shape[1]-1]).reshape((28,28))
        label = df_train.iloc[idx, -1]
        labels.append(label)
        rows_data.append(row_data)


    test_labels = []
    test_rows_data = []
    df_test = pd.read_csv(f'{path_to_train_test_csvs}/test.csv', header=[0])
    for idx in range(len(df_test)):
        test_row_data = np.array(df_test.iloc[idx, :df_test.shape[1]-1]).reshape((28,28))
        test_label = df_train.iloc[idx, -1]
        test_labels.append(test_label)
        test_rows_data.append(test_row_data)

    return np.array(rows_data), np.array(test_rows_data), np.array(labels), np.array(test_labels)


# Utility to warm up the GPU.
def warmup():
    dummy_sample = tf.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    for _ in range(100):
        _ = embedding_model.predict(dummy_sample)


if __name__ == '__main__':
    train_ds, validation_ds = tfds.load(
    "tf_flowers", split=["train[:85%]", "train[85%:]"], as_supervised=True
    )

    images = []
    labels = []

    for (image, label) in train_ds.take(NUM_IMAGES):
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(image.numpy())
        labels.append(label.numpy())

    images = np.array(images)
    labels = np.array(labels)

    # rows_data, test_rows_data, labels, test_labels = get_train_test_dfs("/Users/matt/algorithmic_ml_sandbox/datasets/tf_flowers")

    bit_model = tf.keras.models.load_model("/Users/matt/algorithmic_ml_sandbox/models/vision/flower_model_bit_0.96875")
    bit_model.count_params()
    print(bit_model.layers)

    embedding_model = tf.keras.Sequential(
    [
        tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(scale=1.0 / 255),
        bit_model.layers[1],
        tf.keras.layers.Normalization(mean=0, variance=1),
    ],
        name="embedding_model",
    )

    embedding_model.summary()



    warmup()

    training_files = zip(images, labels)
    lsh_builder = BuildLSHTable(embedding_model)
    lsh_builder.train(training_files)

    # First serialize the embedding model as a SavedModel.
    embedding_model.save("/Users/matt/algorithmic_ml_sandbox/models/vision/embedding_model")

    # Initialize the conversion parameters.
    # params = tf.experimental.tensorrt.ConversionParams(
    #     precision_mode="FP16", maximum_cached_engines=16
    # )

    # Run the conversion.
    # converter = tf.experimental.tensorrt.Converter(
    #     input_saved_model_dir="/Users/matt/algorithmic_ml_sandbox/models/vision/embedding_model", conversion_params=params
    # )
    # converter.convert()
    # converter.save("/Users/matt/algorithmic_ml_sandbox/models/vision/tensorrt_embedding_model")


    # Load the converted model.
    # root = tf.saved_model.load("tensorrt_embedding_model")
    # trt_model_function = root.signatures["serving_default"]


    # warmup()

    # training_files = zip(images, labels)
    # lsh_builder_trt = BuildLSHTable(trt_model_function, concrete_function=True)
    # lsh_builder_trt.train(training_files)

