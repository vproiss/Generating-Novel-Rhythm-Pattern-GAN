"""
This module contains every global variable and global lists

"""

# Ignore drum loops with less onsets than MIN_NB_ONSETS when not much is happening
MIN_NB_ONSETS = 5

# Number of notes per drum loop matrix
NOTES_LENGTH = 32

# The batch_size for the training
BATCH_SIZE = 200

DRUM_CLASSES = [
   'Kick',
   'Snare',
   'Hi-hat closed',
   'Hi-hat open',
   'Tom',
   'Tambourine/Timbale',
   'Cymbal',
   'Percussion',
   'Clap',
]

MIDI_DRUM_MAP = {
     36: 0,
     38: 1,
     40: 1,
     37: 1,
     48: 5,
     50: 6,
     45: 4,
     47: 5,
     43: 4,
     58: 4,
     46: 3,
     26: 3,
     42: 2,
     22: 2,
     44: 2,
     49: 7,
     55: 7,
     57: 7,
     52: 7,
     51: 8,
     59: 8,
     53: 8
 }

DRUM_MIDI_MAP = [ # pianoroll to MIDI - reverse
    36, # 0 Kick / Bass Drum 1
    40, # 1 Snare / Electric Snare
    42, # 2 Hihat Closed
    46, # 3 Hihat Open
    47, # 4 Tom  / Low-mid Tom
    66, # 5 Low Timbale
    51, # 6 Cymbal
    63, # 7 Percussion / Open Hi Conga
    39  # 8 Clap
]

CHOOSEN_GENRE =[
    "rock",
    "funk",
    "latin",
    "jazz",
    "hiphop"
]