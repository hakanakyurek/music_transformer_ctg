import torch

from lib.utilities.constants import *
from lib.utilities.device import get_device

from lib.modules.cocon_block import CoconBlock
from torch.optim.lr_scheduler import LambdaLR

from lib.utilities.lr_scheduling import LrStepTracker
from .music_transformer_base import MusicTransformerBase


# MusicTransformer
class MusicTransformerCoCon(MusicTransformerBase):

    def __init__(self, music_transformer, acc_metric, loss_fn=None, lr=1.0, num_heads=8, d_model=512, dim_feedforward=1024, 
                 max_sequence=2048, keys=True):
        super(MusicTransformerCoCon, self).__init__(acc_metric)
        print('Generatin CoCon model')
        # Key is always the first token if enabled
        self.keys = keys
        self.lr = lr
        self.cs_len = 1
        self.d_model = d_model
        self.loss_fn = loss_fn
        
        self.music_transformer = music_transformer
        self.cocon_block  = CoconBlock(d_model, dim_feedforward, num_heads, max_sequence)
        

    # generate
    def generate(self, primer, context, target_seq_length=1024, temperature=1.0, top_p=0.0, top_k=0):
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
            y = self.music_transformer.Wout(self.forward(gen_seq[..., :cur_i], context[..., :cur_i]))
            y = self.music_transformer.softmax(y / temperature)[..., :TOKEN_END]
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
        hidden_c = hidden_c[:,0,:]
        hidden_c = hidden_c.unsqueeze(1)

        # No start of sentence already
        # History sequence doesn't exist as we aren't doing self supervised learning
        cocon_output = self.cocon_block(hidden_x, context_seq=hidden_c, include_sos_output=True)

        # cocon_lm_tail_input = torch.cat([hidden_x[:, :-1], cocon_output], dim=1)

        cocon_lm_tail_output = self.music_transformer(cocon_output, start_layer=2, 
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

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.cocon_block.parameters(), lr=self.lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
        # opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

        lr_stepper = LrStepTracker(self.d_model, SCHEDULER_WARMUP_STEPS, 1)

        lr_scheduler = LambdaLR(opt, lr_stepper.step)

        return {
        'optimizer': opt, 
        'lr_scheduler': {
            'scheduler': lr_scheduler, 
            'interval': 'step'
        }
    }



