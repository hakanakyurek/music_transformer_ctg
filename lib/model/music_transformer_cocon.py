import torch

from lib.utilities.constants import *
from lib.utilities.device import get_device

from lib.modules.cocon_block import CoconBlock

from .music_transformer import MusicTransformerEncoder


# MusicTransformer
class MusicTransformerCoCon:

    def __init__(self, music_transformer, num_heads=8, d_model=512, dim_feedforward=1024, max_sequence=2048, keys=True):
        super(MusicTransformerCoCon, self).__init__()
        print('Generatin CoCon model')
        # Key is always the first token if enabled
        self.keys         = keys
        self.cs_len = 1
        
        self.music_transformer = music_transformer
        self.cocon_block  = CoconBlock(d_model, dim_feedforward, num_heads, max_sequence)
        

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
            token_probs = y[:, cur_i-1, :]
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

    def forward(self, x, c, mask=True):
        # x input, c context 
        hidden_x = self.music_transformer(x, mask=mask, stop_layer=1)
        hidden_c = self.music_transformer(c, mask=mask, stop_layer=1)

        # No start of sentence already
        # History sequence doesn't exist as we aren't doing self supervised learning
        cocon_output = self.cocon_block(hidden_x, context_seq=hidden_c, include_sos_output=True)

        cocon_lm_tail_input = torch.cat([hidden_x[:, :-1], cocon_output], dim=1)

        cocon_lm_tail_output = self.music_transformer(cocon_lm_tail_input, start_layer=2, 
                                                      stop_layer=self.music_transformer.nlayers)

        return cocon_lm_tail_output
        
    def step(self, batch, acc_metric, pp_metric):
        c, x, tgt = batch

        y = self.music_transformer.Wout(self.forward(x, c))

        pp_metric.update(y, tgt)

        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt = tgt.flatten()

        loss = self.loss_fn.forward(y, tgt)

        acc_metric.update(y, tgt)

        return loss, y





