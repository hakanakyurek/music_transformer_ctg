import torch

from lib.utilities.constants import *
from lib.utilities.device import get_device

from .music_transformer import MusicTransformerEncoder

def repetition_penalty(logits, penalty_factor):
    """
    Penalizes repeated tokens in the logits tensor by adding a penalty factor
    to the logits of tokens that have already been generated in the sequence.
    """
    batch_size, sequence_length, vocab_size = logits.shape
    previous_tokens = torch.argmax(logits, dim=-1)
    for i in range(batch_size):
        for j in range(1, sequence_length):
            if previous_tokens[i, j] == previous_tokens[i, j-1]:
                logits[i, j, previous_tokens[i, j]] /= penalty_factor
    return logits

# MusicTransformer
class MusicTransformerCTRL(MusicTransformerEncoder):

    def __init__(self, loss_fn=None, acc_metric=None, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False, lr=1.0, keys=False):
        super(MusicTransformerCTRL, self).__init__(loss_fn, acc_metric, n_layers, num_heads, d_model,
                                                   dim_feedforward, dropout, max_sequence, rpr, lr)
        print('Generatin CTRL model')
        # Key is always the first token if enabled
        self.keys         = keys

    # generate
    def generate(self, primer=None, target_seq_length=1024, temperature=1.0, top_p=0.0, top_k=0):
        """
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
        """

        assert (not self.training), "Cannot generate while in training mode"

        print(f"Generating sequence of max length: {target_seq_length}")

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())


        # print("primer:",primer)
        # print(gen_seq.shape)
        # Here cur_i is the current token index
        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            y = self.Wout(self.forward(gen_seq[..., :cur_i]))
            y = self.softmax(y / temperature)[..., :TOKEN_END]
            y = repetition_penalty(y, 2)

            token_probs = y[:, cur_i-1, :]
            # token_probs = torch.topk(token_probs, 8)
            # next_token = torch.argmax(token_probs)
            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            # print("next token:",next_token)
            gen_seq[:, cur_i] = next_token


            # Let the transformer decide to end if it wants to
            if(next_token == TOKEN_END):
                print(f"Model called end of sequence at: {cur_i}/{target_seq_length}")
                break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(f"{cur_i}/{target_seq_length}")

        return gen_seq[:, :cur_i]
