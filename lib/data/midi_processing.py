import torch
from lib.utilities.constants import TOKEN_END, TOKEN_START, TORCH_LABEL_TYPE, TOKEN_PAD
import random


SEQUENCE_START = 0

# process_midi
def process_midi(raw_mid, max_seq, random_seq, key=None):
    """

    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.

    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)
    tgt = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    # Shift to the right by one
    if(raw_len < full_seq):
        x[:raw_len]         = raw_mid
        tgt[:raw_len-1]     = raw_mid[1:]
        tgt[raw_len-1]      = TOKEN_END
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]


    # print("x:",x)
    # print("tgt:",tgt)

    return x, tgt

# process_midi for ed arch
def process_midi_ed(raw_mid, max_seq, random_seq):
    """

    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.

    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)
    tgt_input = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)
    tgt_output = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq for ed arch

    if(raw_len == 0):
        return x, tgt_input, tgt_output

    # Shift to the right by one
    if(raw_len < full_seq):
        x[:raw_len] = raw_mid
        
        tgt_input[0] = TOKEN_START
        tgt_input[1:raw_len] = raw_mid[1:raw_len]

        tgt_output[:raw_len - 1] = raw_mid[:raw_len - 1]
        tgt_output[-1] = TOKEN_END
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        
        tgt_input[0] = TOKEN_START
        tgt_input[1:full_seq - 1] = data[1:full_seq - 1]

        tgt_output[:full_seq - 2] = data[:full_seq - 2]
        tgt_output[-1] = TOKEN_END


    # print("x:",x)
    # print("tgt:",tgt)

    return x, tgt_input, tgt_output
