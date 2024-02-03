"""
This module contains all preprocessing functions
contains:
    duplicate_multiple_style(data)
    get_pianomatrices_of_drums(midi_file, binary)
    data_processing(data)

"""
import pandas as pd
import numpy as np
import tensorflow as tf
import pretty_midi

from util import MIDI_DRUM_MAP, DRUM_CLASSES, MIN_NB_ONSETS, NOTES_LENGTH, BATCH_SIZE


def duplicate_multiple_styles(data):
    """ This functions duplicates every data entry, that hast more than one style. It
    returns multiple data entries, one for each music style, containing only one style.
    If the data entry just has one style it just return it

    :param
        data: a pandas dataframe entry containing a "stlye" column
    :return:
        duplicated_datas(pd.DataFrame) DataFrame with one or multiple entries
    """
    duplicated_datas = pd.DataFrame()
    styles = data.style.split("/")
    for style in styles:
        duplicated_data = data
        duplicated_data["style"] = style
        duplicated_datas = duplicated_datas.append(duplicated_data, ignore_index=True)
    return duplicated_datas


def get_pianomatrices_of_drums(midi_file, binary="False"):
    """
    Loads a midi_file and converts it into a drum matrix.

    :param
        midi_file (str): a filepath as string to the midi file, ends with .mid
        binary (boolean): determine if the returning matrix is binary or float
    :return:
        drum_matrices (np.array): Array with onsets and velocity of the midi file
        size = (xx, notes_length, drumclasses)
    """

    number_of_different_notes = len(DRUM_CLASSES)  # reduction of less note values


    # load midi file
    try:
        pm = pretty_midi.PrettyMIDI(midi_file=midi_file)
    except:
        print("Failed to load midi: ", midi_file)
        return None

    # get timing of notes in absolut seconds
    beats = pm.get_beats()

    # make it to 16th notes
    beats_ticks = []  # converting to ticks
    for i in range((len(beats) + 1) * 4):
        beats_ticks.append(i * pm.resolution / 4)

    num_notes16 = len(beats_ticks)  # Number of all notes

    drum_matrices = []  # create pianoroll matrix (resolution: 16th note))

    missing_notes = []  # list for not found notes

    for instrument in pm.instruments:
        # splits the whole song into smaller hops
        for hops in range(0, num_notes16 - 16, 16):  # hopsize = 1bar = 16 16th notes
            drum_matrix = np.zeros((NOTES_LENGTH, number_of_different_notes), dtype='float')

            for note in instrument.notes:
                idx_start = (np.abs(beats_ticks - pm.time_to_tick(note.start)).argmin())

                if hops <= idx_start < hops + NOTES_LENGTH:
                    if note.pitch in MIDI_DRUM_MAP:
                        drum_note = MIDI_DRUM_MAP[note.pitch]

                        # differentiate between binary and float outcome
                        if binary:
                            drum_matrix[idx_start - hops, drum_note] = 1
                        # relative velocity/intensity of the note
                        else:
                            drum_matrix[idx_start - hops, drum_note] = note.velocity / 127.

                    else:
                        if note.pitch not in missing_notes:
                            missing_notes.append(note.pitch)

            if np.sum(drum_matrix > 0.) >= MIN_NB_ONSETS:
                # ignore the last part of the midi file where rhythm ends in the first bar
                if np.sum(drum_matrix[NOTES_LENGTH // 2:, :]) > 0:
                    drum_matrices.append(drum_matrix)

    if missing_notes:
        print(f"The following notes aren't initialized {missing_notes}")

    return np.array(drum_matrices)


def data_processing(data):
    """ This function preprocesses the data sets and turn it into a batched, prefetched
        tensorlow dataset

    :param data (np.array): A numpy array, containing drum matrices
    :return (tf.data.Dataset): the processed dataset
    """
    # from tensor dataset
    data = tf.data.Dataset.from_tensor_slices(data)

    # change datatype to float 32
    data = data.map(lambda matrix: (tf.cast(matrix, tf.float32)))
    # shuffle the datasets
    data = data.shuffle(buffer_size=1000)
    # batch the datasets
    data = data.batch(BATCH_SIZE)
    # prefetch the datasets
    data = data.prefetch(200)

    return data
