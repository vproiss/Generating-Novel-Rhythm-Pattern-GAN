"""
This is the class of the Generator, for the GAN model

"""
import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense, LeakyReLU, Reshape
from util import NOTES_LENGTH, DRUM_CLASSES

sequence_length = NOTES_LENGTH  # The length of the incoming drum matrices sequences

nb_notes = len(DRUM_CLASSES)  # Number of possible notes


class Generator(tf.keras.Model):
    """Generator part of the GAN model"""

    def __init__(self, optimizer=tf.keras.optimizers.RMSprop(lr=0.0004)):
        """ Initializer

        :param optimizer: (tf.keras.optimizers.Optimizer) optimizer to use in training,
                        default RMSprop with learning rate 0.0004.
        """
        super(Generator, self).__init__()

        self.optimizer = optimizer

        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.input_layer = Input(shape=(sequence_length*nb_notes,))
        self.all_layers = [
            Dense(1024),
            LeakyReLU(alpha=0.2),
            Reshape((32,32)),
            LSTM(512, return_sequences=True, activation="tanh"),
            LSTM(512, return_sequences=True, activation="tanh"),
            LSTM(9, return_sequences=True, activation="sigmoid")
        ]
        self.out = self.call(self.input_layer, training=True)

    def call(self, x, training):
        """ Call function of the model. Propagates input through the layers

        :param x: (tensor) input to the model
        :param training: (boolean) whether to use training or inference mode, default: False (inference)
        :return: (tensor) tensor x after running through the model
        """
        for layer in self.all_layers:
            try:
                x = layer(x, training)
            except:
                x = layer(x)
        return x
