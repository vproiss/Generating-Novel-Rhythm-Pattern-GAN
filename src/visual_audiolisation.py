"""
This module contains the function to visualize or audiolize the drum matrices:
contains:
    plot_drum_matrix(matrix)
    play_drum_matrix(matrix, tempo):
    get_audio_from_matrix(matrix, tempo):

"""

import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

from util import DRUM_CLASSES, DRUM_MIDI_MAP
from IPython.display import Audio


def plot_drum_matrix(matrix):
    """This function gets a drum matrix and plots it in a headmap

    :param matrix: (np.array) A drum matrix
    """
    if matrix is not None:
        matrix = np.transpose(np.squeeze(matrix))
        plt.matshow(matrix)
        plt.show()


def play_drum_matrix(matrix, tempo=120):
    """ Receives a drum matrix and a tempo feed them into the get_audio_from_matrix
    where it is translated into a MIDI file and a playable audio data and convert it
    into a wave file

    :param matrix: (np.array) Matrix with 9 columns, each for all DRUM_CLASSES
    :param tempo: (int) The tempo for the resulting audio file
    :return: (audio_data)
    """
    audio_data = get_audio_from_matrix(matrix, tempo=tempo)
    Audio(audio_data, rate=44100)
    return audio_data


def get_audio_from_matrix(matrix, tempo=120):
    """ Receives a drum matrix and transform every matrix entry into a corresponding
    midi note and combines them to a MIDI instrument.

    :param matrix: (np.array) Matrix with 9 columns, each for all DRUM_CLASSES
    :param tempo: (int) The tempo for the resulting MIDI file
    :return audio_data: (audio_data) A displayable audio data
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)  # midi_object
    pm_inst = pretty_midi.Instrument(0, is_drum=True)  # midi_instrument

    timestep = (60./tempo) / 4  # 16th notes
    for position, timeslot in enumerate(matrix):
        for inst, onset in enumerate(timeslot):
            if onset > 0:
                note_number = DRUM_MIDI_MAP[inst]  # drum sound
                velocity = int(onset * 127.)  # velocity/intensity of the note
                start = timestep * position  # note start point
                end = timestep * (position + 0.5)  # note end point

                # create the midi note
                note = pretty_midi.Note(velocity=velocity, pitch=note_number, start=start, end=end)
                pm_inst.notes.append(note)
    pm.instruments.append(pm_inst)

    # midi -> audio
    audio_data = pm.fluidsynth()
    return audio_data
