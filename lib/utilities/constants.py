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

VOCAB_SIZE_KEYS         = TOKEN_KEYS + 1
VOCAB_SIZE_NORMAL       = TOKEN_PAD + 1

vocab = {
    'size': -1
}

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4

TASK = ''
EXPERIMENT = ''
