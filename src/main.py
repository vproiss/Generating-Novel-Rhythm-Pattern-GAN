"""
This is the Main file for our Final project. This file should be run to start the data
processing and training loops.
Run with flags:
-e --epochs (int): Number of training epochs
-v --visualize (bool): Visualise drum matrices between training epochs
-t --second-training: enables the second training process

"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from mlflow import log_param, log_metric, set_tracking_uri
from util import BATCH_SIZE
from Generator import Generator
from Discriminator import Discriminator
from training_loop import training_loop, genre_training_loop
from preprocessing_functions import data_processing
from visual_audiolisation import plot_drum_matrix, play_drum_matrix
from scipy.io import wavfile

# Add arguments for running on grid
parser = argparse.ArgumentParser(description="Main training file")
parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs", default=10)
parser.add_argument("-v", "--visualize", action="store_true", help="visualize drum matrices between epochs", default=False)
parser.add_argument("-t", "--second-training", action="store_true", help="Let the Genre training part run", default=True)
parser.add_argument("--RMSProp", type=float, help="Use Optimizer RMSProp learning rate x for training", default=None)
parser.add_argument("--SGD", type=float, help="Use Optimizer Stochastic gradient descent learning rate x for training", default=None)
parser.add_argument("--adam", type=float, help="Use Adam descent learning rate x for training", default=None)
parser.add_argument("--log_folder", help="Where to log the mlflow results", default="../data/mlflow")
args = parser.parse_args()

# setting log folder for mlflow grid data
set_tracking_uri(args.log_folder)

# Provide dataset
if not os.path.exists("../data/cleaned_data.pkl"):
    print("No .pickle file found, dataset will be created:")
    try:
        os.system("python data_processing.py")
        dataset = pd.read_pickle("../data/cleaned_data.pkl")
    except:
        print("Could not execute the data_processing.py file")
else:
    dataset = pd.read_pickle("../data/cleaned_data.pkl")

# makes every drum matrix to a single training point
data = np.vstack(dataset["drum_matrices"])
data = data_processing(data)

# Choose Optimizer
if args.RMSProp is not None:
    print(f"   Training with RMSProp and LR = {args.RMSProp}")
    log_param("RMSProp", args.RMSProp)  # mlflow logs
    optimizer = tf.keras.optimizers.RMSprop(lr=args.RMSProp)

elif args.SGD is not None:
    print(f"   Training with SGD and LR = {args.SGD}")
    log_param("SGD", args.SGD)  # mlflow logs
    optimizer = tf.keras.optimizers.SGD(lr=args.SGD)

elif args.adam is not None:
    print(f"   Training with Adam and LR = {args.adam}")
    log_param("Adam", args.adam)  # mlflow logs
    optimizer = tf.keras.optimizers.Adam(lr=args.adam)

# Initialise generator and discriminator
generator = Generator(optimizer=optimizer)
discriminator = Discriminator(optimizer=optimizer)

# The first training process
print(f"First training process")
gen_loss, disc_loss, disc_acc, beats = training_loop(data, generator, discriminator,
                                                     BATCH_SIZE, epochs=args.epochs,
                                                     visualize=args.visualize)
# Save the results in mlflow
log_param("Generator Loss", gen_loss)
log_param("Discriminator Loss", disc_loss)
log_param("Discriminator Accuracy", disc_acc)
log_param("Epoch Beats", beats)

# Save one drum matrix for mlflow
fake_beat = generator(tf.random.normal(shape=(1, 288)), training=False)
log_param("Generated Beat", fake_beat[0])

print(f"Mlflow data saved:")

# Create Audio data
audio_data = play_drum_matrix(fake_beat[0])
# Save Models (sadly dont work) and wav files
if args.RMSProp is not None:
    #generator.save(f"../data/models/generator/RMSProp{str(args.RMSProp)}")
    #discriminator.save(f"../data/models/discriminator/RMSProp{str(args.RMSProp)}")
    wavfile.write(f"../data/beats/RMSProp{args.RMSProp}.wav", 44100, audio_data)

elif args.SGD is not None:
    #generator.save(f"../data/models/generator/SGD{str(args.SGD)}")
    #discriminator.save(f"../data/models/discriminator/SGD{str(args.SGD)}")
    wavfile.write(f"../data/beats/SGD{args.SGD}.wav", 44100, audio_data)

elif args.adam is not None:
    #generator.save(f"../data/models/generator/Adam{str(args.adam)}")
    #discriminator.save(f"../data/models/discriminator/Adam{str(args.adam)}")
    wavfile.write(f"../data/beats/Adam{args.adam}.wav", 44100, audio_data)


# Second training loop for rock genre
if args.second-training:
    # Split into rock and not rock styles
    genre_data = dataset[dataset["style"] == "rock"]
    other_data = dataset[dataset["style"] != "rock"]

    genre_data = np.vstack(genre_data["drum_matrices"])
    other_data = np.vstack(other_data["drum_matrices"])

    # Length of the data
    len_g = len(genre_data)
    len_o = len(other_data)

    # make both dataset the same length
    if len_o <= len_g:
        for _ in range(np.abs(len_o - len_g)):
            other_data = np.append(other_data, np.expand_dims(other_data[np.random.choice(range(len_o))], axis=0), axis=0)
    if len_o > len_g:
        for _ in range(np.abs(len_o - len_g)):
            genre_data = np.append(genre_data, np.expand_dims(genre_data[np.random.choice(range(len_g))], axis=0), axis=0)

    # preprocessing
    genre_data = data_processing(genre_data)
    other_data = data_processing(other_data)

    # Trains the discriminator for just one genre
    print("\nGenre Discrimination Training")
    disc2_loss, disc2_acc = genre_training_loop(genre_data, other_data, discriminator, BATCH_SIZE,
                                                epochs=args.epochs)
    # Save the results in mlflow
    log_param("Discriminator2 Loss", disc2_loss)
    log_param("Discriminator2 Accuracy", disc2_acc)

    # Trains the generator for the rock style
    print("\nGenre Rock Discriminator")
    gen3_loss, disc3_loss, disc3_acc, beats3 = training_loop(genre_data, generator, discriminator,
                                                             BATCH_SIZE, epochs=args.epochs,
                                                             visualize=args.visualize)
    # Save the results in mlflow
    log_param("Rock Generator Loss", gen3_loss)
    log_param("Rock Discriminator Loss", disc3_loss)
    log_param("Rock Discriminator Accuracy", disc3_acc)
    log_param("Epoch Rock Beats", beats3)
    # Create one final rock beat
    final_beat = generator(tf.random.normal(shape=(1, 288)), training=False)
    log_param("Final Beat", final_beat)
    # Plot one final beat
    plot_drum_matrix(final_beat.numpy())
    # Form wav file
    audio_data = play_drum_matrix(final_beat[0])
    wavfile.write("../data/beats/final_rock_beat.wav", 44100, audio_data)
