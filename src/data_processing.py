"""
This module processes the data. The data for training only should contain the
beat "beat_type" not fill, because the fills are way too short. As well all unnecessary
information are dropped.

Is called by main.py to download and process the data.
"""

import os.path
import pandas as pd

from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm

from util import CHOOSEN_GENRE, MIN_NB_ONSETS
from preprocessing_functions import duplicate_multiple_styles, get_pianomatrices_of_drums


# Download datasets
if not os.path.exists("../data/groove"):
    print("Download started")
    http_response = urlopen("https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip")
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path="../data")
    print("Download finished")
else:
    print("Load file from local")

# Read the CSV file
dataset = pd.read_csv("../data/groove/info.csv")

dataset_cleaned = pd.DataFrame()

# duplicates all songs that fit into multiple styles
for _, row in dataset.iterrows():
    dataset_cleaned = dataset_cleaned.append(duplicate_multiple_styles(row))

# remove all midi files that are not long enough
dataset_cleaned = dataset_cleaned[dataset_cleaned.duration > MIN_NB_ONSETS]
#dataset_cleaned = dataset_cleaned[dataset_cleaned.beat_type != "fill"] # another way to filter the short beats

# just keep the filepath and style
dataset_cleaned = dataset_cleaned[["style", "midi_filename"]]

# just keep the styles with the most songs
print(f"Uses 5 most styles:{CHOOSEN_GENRE}")
dataset_cleaned = dataset_cleaned[dataset_cleaned["style"].isin(CHOOSEN_GENRE)]

# add correct file path to filename
dataset_cleaned["midi_filename"] = "../data/groove/" + dataset_cleaned["midi_filename"]

# translates the midi file into a drum matrix, takes a while
print("Start translating midi into drum matrix:")
dataset_cleaned["drum_matrices"] = [get_pianomatrices_of_drums(midi_file, False) for midi_file
                                    in tqdm(dataset_cleaned["midi_filename"])] # tqdm visualise progress
print("Translation finished!")

# save cleaned_data as pickle file for later use
dataset_cleaned.to_pickle("../data/test.pkl")
