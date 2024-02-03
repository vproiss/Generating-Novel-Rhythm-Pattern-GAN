"""
This is the class of the Discriminator, for the GAN model

"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, Dense, LSTM, Reshape, BatchNormalization, Flatten, LeakyReLU

from util import NOTES_LENGTH, DRUM_CLASSES, BATCH_SIZE

sequence_length = NOTES_LENGTH  # The length of the incoming drum matrices sequences

nb_notes = len(DRUM_CLASSES)  # Number of possible notes

droprate = 0.4


class Discriminator(tf.keras.Model):
    """Discriminator part of the GAN model"""

    def __init__(self, optimizer=tf.keras.optimizers.RMSprop(lr=0.0004)):
        """ Initializer

        :param optimizer: (tf.keras.optimizers.Optimizer) optimizer to use in training,
                        default RMSprop with learing rate 0.0004.
        """
        super(Discriminator, self).__init__()

        self.optimizer = optimizer

        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.input_layer = Input(shape=(sequence_length, nb_notes,), batch_size=BATCH_SIZE)
        self.all_layers = [
            Bidirectional(LSTM(64, return_sequences=True, activation="tanh",
                               dropout=droprate, recurrent_dropout=droprate)),
            Bidirectional(LSTM(64, return_sequences=False, activation="tanh",
                               dropout=droprate, recurrent_dropout=droprate)),
            Dense(units=1, activation="sigmoid")
        ]

        self.out = self.call(self.input_layer, training=True)

    def call(self, x, training=False):
        """ Call function of the model. Propagates input through the layers

        :param x: (tensor) input to the model
        :param training: (boolean) whether to use training or inference mode, default: False (inference)
        :return: (tensor) tensor x after running through the model
        """
        for layers in self.all_layers:
            try:
                x = layers(x, training=training)
            except:
                x = layers(x)
        return x
