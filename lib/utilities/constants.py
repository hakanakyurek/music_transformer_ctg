import torch

from lib.midi_processor.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF, RANGE_VEL, RANGE_TIME_SHIFT

SEPERATOR               = "========================="

# Taken from the paper
ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9

LR_DEFAULT_START        = 1.0
SCHEDULER_WARMUP_STEPS  = 4000
# LABEL_SMOOTHING_E       = 0.1

# DROPOUT_P               = 0.1

TOKEN_START             = -1
TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD               = TOKEN_END + 1

RANGE_KEYS              = 24 # major and minor keys in circle of fifths
TOKEN_KEYS              = TOKEN_PAD + RANGE_KEYS

TOKEN_TRUE              = TOKEN_KEYS + 1
TOKEN_FALSE             = TOKEN_TRUE + 1

KEY_DICT = {
    'c# minor': 1,
    'F major': 2,
    'D major': 3, 
    'E major': 4, 
    'B major': 5, 
    'g minor': 6, 
    'e minor': 7, 
    'F# major': 8, 
    'd minor': 9, 
    'f# minor': 10, 
    'C major': 11, 
    'g# minor': 12, 
    'A major': 13, 
    'b- minor': 14, 
    'b minor': 15, 
    'A- major': 16, 
    'E- major': 17, 
    'G major': 18, 
    'f minor': 19, 
    'B- major': 20, 
    'c minor': 21, 
    'e- minor': 22, 
    'a minor': 23, 
    'C# major': 24}

KEY_VOCAB = {}

for k in KEY_DICT.keys():
    KEY_VOCAB[k] = KEY_DICT[k] + TOKEN_PAD

VOCAB_SIZE_KEYS_GEDI    = TOKEN_FALSE + 1
VOCAB_SIZE_KEYS         = TOKEN_KEYS + 1
VOCAB_SIZE_NORMAL       = TOKEN_PAD + 1

vocab = {
    'size': -1
}

ARTIST_DICT = {
    'Joseph Haydn': 0,
    'Felix Mendelssohn': 1,
    'Johann Strauss': 2,
    'Mikhail Glinka': 3,
    'Domenico Scarlatti': 4,
    'Antonio Soler': 5,
    'Leoš Janáček': 6,
    'César Franck': 7,
    'Jean-Philippe Rameau': 8,
    'Muzio Clementi': 9,
    'Johann Sebastian Bach': 10,
    'Claude Debussy': 11,
    'Johannes Brahms': 12,
    'Johann Christian Fischer': 13,
    'Nikolai Rimsky-Korsakov': 14,
    'Nikolai Medtner': 15,
    'Carl Maria von Weber': 16,
    'Niccolò Paganini': 17,
    'Franz Liszt': 18,
    'Sergei Rachmaninoff': 19,
    'Percy Grainger': 20,
    'Georges Bizet': 21,
    'Mily Balakirev': 22,
    'Franz Schubert': 23,
    'George Frideric Handel': 24,
    'Orlando Gibbons': 25,
    'Robert Schumann': 26,
    'Edvard Grieg': 27,
    'Fritz Kreisler': 28,
    'Wolfgang Amadeus Mozart': 29,
    'Giuseppe Verdi': 30,
    'Pyotr Ilyich Tchaikovsky': 31,
    'Richard Wagner': 32,
    'Modest Mussorgsky': 33,
    'Alexander Scriabin': 34,
    'Alban Berg': 35,
    'Johann Pachelbel': 36,
    'George Enescu': 37,
    'Henry Purcell': 38,
    'Frédéric Chopin': 39,
    'Ludwig van Beethoven': 40,
    'Isaac Albéniz': 41,
    'Charles Gounod': 42
}

GENRE_DICT = {
    'rap': 0, 
    'latin': 1, 
    'unknown': 2, 
    'electronic': 3, 
    'religious': 4, 
    'reggae': 5, 
    'pop': 6, 
    'ambient': 7, 
    'country': 8, 
    'classical': 9, 
    'rock': 10, 
    'soundtracks': 11, 
    'soul': 12, 
    'folk': 13, 
    'blues': 14, 
    'children': 15, 
    'jazz': 16, 
    'world': 17}

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4

TASK = ''
EXPERIMENT = ''
