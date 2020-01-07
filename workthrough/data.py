# set the environment path to find Recommenders
import sys
sys.path.append("../../")

import pyspark
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from reco_utils.common.spark_utils import start_or_get_spark
from reco_utils.dataset.download_utils import maybe_download
from reco_utils.dataset.python_splitters import (
    python_random_split,
    python_chrono_split,
    python_stratified_split
)
from reco_utils.dataset.spark_splitters import (
    spark_random_split,
    spark_chrono_split,
    spark_stratified_split,
    spark_timestamp_split
)

print("System version: {}".format(sys.version))
print("Pyspark version: {}".format(pyspark.__version__))

#%%

DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
DATA_PATH = "ml-100k.data"

# Set the column names that will be imported
COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_PREDICTION = "Rating"
COL_TIMESTAMP = "Timestamp"

filepath = maybe_download(DATA_URL, DATA_PATH)

data = pd.read_csv(filepath, sep="\t", names=[COL_USER, COL_ITEM, COL_RATING, COL_TIMESTAMP])

# Head of pandas lets you look at some initial instances
print(data.head())
# Describe shows data statistics of each column
print(data.describe())

# Print the number of users, items and ratings for the chosen dataset
print(
    "Total number of ratings are\t{}".format(data.shape[0]),
    "Total number of users are\t{}".format(data[COL_USER].nunique()),
    "Total number of items are\t{}".format(data[COL_ITEM].nunique()),
    sep="\n"
)


# Change the format of the timestamp column
orig_function = data.apply(
    lambda x: datetime.strftime(datetime(1970, 1, 1, 0, 0, 0) + timedelta(seconds=x[COL_TIMESTAMP].item()), "%Y-%m-%d %H:%M:%S"),
    axis=1
)

print(data.head(orig_function))

iso_8601_python = data.apply(
    lambda x: x[COL_TIMESTAMP].item().isoformat(),
    axis=1
)

print(data.head(iso_8601_python))

data[COL_TIMESTAMP] = data.apply(
    lambda x: x[COL_TIMESTAMP].item().isoformat(),
    axis=1
)


#%%

data.head()

# Stratified split the data
data_train, data_test = python_stratified_split(
    data, filter_by="user", min_rating=10, ratio=0.7,
    col_user=COL_USER, col_item=COL_ITEM
)
