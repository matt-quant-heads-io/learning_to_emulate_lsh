import argparse
import logging
import math
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_LOADER_FUNCTION_DICT = {
    "tf_flowers": {
        "name": "tf_flowers",
        "path": "~/algorithmic_ml_sandbox/datasets/tf_flowers",
        "num_columns": 784,
    },
    # "tf_text": {
    #     "name": "tf_flowers",
    #     "path": "../datasets/tf_flowers"
    # }
}

IMAGE_SIZE = 28
NUM_IMAGES = 1000


def initialize_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s> %(filename)s::%(funcName)s::%(lineno)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# def get_parse_args():
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument(
#         "--dataset", required=True, choices=["tf_flowers","mnist", "fashion_mnist", "cifar10"]
#     )

#     args = argparser.parse_args()

#     return args


def return_mongo_client_and_create_database_if_does_not_exist(mongo_db_name):
    try:
        mongo_db_adapter = lib.adapters.mongo_adapter.MongoAdapter()

        mongo_dbs_that_exist_list = mongo_db_adapter.client.list_database_names()
        if mongo_db_name in mongo_dbs_that_exist_list:
            print("The database exists.")

        # mongo_db_client = (
        #     mongo_db_adapter.client
        # )  # TODO: refactor to use this .get_mongo_client() instead of .client

        return mongo_db_adapter
    except RuntimeError as e:
        msg = "Error in return_mongo_client_and_create_database_if_does_not_exist"
        raise e(msg)


def compute_distance_between_two_images(curr_image, other_image):
    num_rows, num_cols = len(curr_image), len(curr_image[0])
    absolute_difference = 0

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            absolute_difference += abs(
                curr_image[row_idx][col_idx] - other_image[row_idx][col_idx]
            )

    return absolute_difference


def compute_distances_from_curr_images(curr_img_index, test_dataset, train_dataset):
    absolute_differences = []
    curr_image = list(test_dataset)[curr_img_index]

    for img_index, (img, _) in enumerate(train_dataset):
        absolute_differences.append(
            compute_distance_between_two_images(curr_image[0].numpy(), img.numpy())
        )

    return absolute_differences


def get_ids_of_closest_images(curr_img_index, test_dataset, train_dataset):
    distances = compute_distances_from_curr_images(
        curr_img_index, test_dataset, train_dataset
    )
    min_distance = min(distances)
    ids_of_most_similar_images = []

    for idx, distance in enumerate(distances):
        if distance == min_distance:
            ids_of_most_similar_images.append(idx)

    return ids_of_most_similar_images


def main():
    for _, ds_vals_dict in DATASET_LOADER_FUNCTION_DICT.items():
        ds_name = ds_vals_dict["name"] 
        train_path = f'{ds_vals_dict["path"]}/train.csv'
        test_path = f'{ds_vals_dict["path"]}/test.csv'


        train_ds, test_ds = tfds.load("tf_flowers", split=["train[:85%]", "train[85%:]"], as_supervised=True)


        train_dict = {str(i):[] for i in range(ds_vals_dict['num_columns'])}
        train_dict["label"] = []


        for (image, label) in train_ds.take(len(train_ds)):
            image = tf.image.rgb_to_grayscale(
                image, name=None
            )
            image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            flat_image = image.numpy().flatten()
            for i in range(len(flat_image)):
                train_dict[str(i)].append(flat_image[i])
            
            train_dict["label"].append(label.numpy())
            

        df_train = pd.DataFrame(train_dict)
        df_train.to_csv(train_path, index=False)

        logging.info(f"Wrote train data to {train_path}")


        test_dict = {str(i):[] for i in range(ds_vals_dict['num_columns'])}
        test_dict["label"] = []

        for (image, label) in test_ds.take(len(test_ds)):
            image = tf.image.rgb_to_grayscale(
                image, name=None
            )
            image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            flat_image = image.numpy().flatten()
            for i in range(len(flat_image)):
                test_dict[str(i)].append(flat_image[i])
            
            test_dict["label"].append(label.numpy())


        df_test = pd.DataFrame(test_dict)
        df_test.to_csv(test_path, index=False)

        logging.info(f"Wrote test data to {test_path}")



if __name__ == "__main__":
    initialize_logging()
    main()
