import torch.nn as nn
import torch
import math

from .conv1d import *
from torch.nn import Linear, Dropout
from torch.nn import functional as F


class CoconBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_heads=8, max_sequence=2048, 
                 dropout=0.1, output_attn=False, scale=False):
        super().__init__()

        self.ln_1 = nn.LayerNorm(d_model)

        self.sos_h = nn.Parameter(torch.zeros(d_model))
        self.mask_h = nn.Parameter(torch.zeros(d_model))

        self.cocon_attn = CoconAttention(d_model, num_heads, max_sequence, output_attn, scale)
        self.ln_2 = nn.LayerNorm(d_model)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.instance_norm = nn.InstanceNorm1d(d_model, affine=False, track_running_stats=False)

        self.attn_dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def forward(self, x, context_seq=None, history_seq=None, layer_past=None, 
                attention_mask=None, head_mask=None, 
                include_sos_output=False, cs_masked_indices=None,
                tis_masked_indices=None, cs_self_attn_mask_prob=0, 
                context_attn_bias=0, context_seq_len_list=None):
        
        if cs_masked_indices is not None and context_seq is not None:
            context_seq = context_seq.clone() # avoid overwrite original context_seq with mask_h
            context_seq[cs_masked_indices] = self.mask_h

        if tis_masked_indices is not None and x is not None:
            x = x.clone() # avoid overwrite original x with mask_h
            x[tis_masked_indices] = self.mask_h

        if history_seq is not None:
            history_seq_len = history_seq.shape[1]
            if x is not None:
                cocon_attn_input = torch.cat([history_seq, x], dim=1)
            else:
                cocon_attn_input = history_seq
        elif x is not None:
            history_seq_len = 0
            # batch_size = x.shape[0]
            # sos_h = self.sos_h.view(1, 1, -1).expand(batch_size, -1, -1)
            # cocon_attn_input = torch.cat([sos_h, x], dim=1)
            cocon_attn_input = x

        x = cocon_attn_input

        cocon_attn_input_ln_1 = self.ln_1(cocon_attn_input)
        x_1_output = cocon_attn_input_ln_1

        output_attn = self.cocon_attn(
            x_1_output, context_seq, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, cs_self_attn_mask_prob=cs_self_attn_mask_prob, history_seq_len=history_seq_len, 
            context_attn_bias=context_attn_bias, context_seq_len_list=context_seq_len_list
        )
        a = output_attn[0]  # output_attn: (a), present, (attentions)
        # H^L_preconv
        x = x + a

        # Skip history_seq computation if history_seq_len > 1
        if history_seq_len > 1:
            x = x[:, history_seq_len-1:]


        x_ln_2 = self.ln_2(x)
        x_2_output = x_ln_2
        m = self.linear2(self.dropout(F.silu(self.linear1(x_2_output))))
        # H^L
        x = x + m

        if include_sos_output:
            cocon_output = x
        else:
            cocon_output = x[:, 1:, :]

        return cocon_output

    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CoconAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, max_sequence=2048, dropout=0.1, output_attn=False, scale=False):
        super().__init__()
        self.output_attentions = output_attn

        n_state = d_model  # in Attention: n_state=768 (nx=n_embd)
        n_ctx = max_sequence
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % num_heads == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

        self_token_mask = torch.ones(n_ctx, n_ctx)
        self_token_mask.fill_diagonal_(0)
        self.register_buffer("self_token_mask", self_token_mask.view(1, 1, n_ctx, n_ctx))
        self.n_head = num_heads
        self.split_size = n_state
        self.scale = scale

        self.ref_source_attn = Conv1D(n_state * 2, d_model)
        self.c_attn = Conv1D(n_state * 3, d_model) # input has dim of nx
        self.c_proj = Conv1D(n_state, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, cs_self_attn_mask_prob=0, history_seq_len=None, context_seq_present=True, context_seq_len=0, context_attn_bias=0, context_seq_len_list=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd : ns, :ns]
        w = w * b - 1e4 * (1 - b)

        # self_token_mask computation
        if cs_self_attn_mask_prob > 0 and context_seq_present:
            if history_seq_len == 0:
                history_seq_offset = 0
            else:
                history_seq_offset = history_seq_len - 1
            self_token_mask = self.self_token_mask[:, :, :nd, history_seq_offset:history_seq_offset+ns]
            self_token_mask = self_token_mask.repeat(w.shape[0],1,1,1)

            if cs_self_attn_mask_prob != 1:
                # compute unmasked indices
                self_token_unmask_prob = 1 - cs_self_attn_mask_prob
                unmask_prob_matrix = torch.full(self_token_mask.shape[:-1], self_token_unmask_prob)
                unmasked_indices = torch.bernoulli(unmask_prob_matrix).bool()
                self_token_mask[unmasked_indices] = 1

            w = w * self_token_mask - 1e4 * (1 - self_token_mask)
            
        
        if context_attn_bias != 0:
            if context_seq_len_list is None:
                context_attn_bias_mask = torch.ones(w.shape) # N, H, Q, V
                context_attn_bias_mask[:,:,:, :context_seq_len] = 0
                context_attn_bias_mask = context_attn_bias_mask.to(w.device)
                w = w + context_attn_bias * (1 - context_attn_bias_mask)     
            else:
                current_context_start_ind = 0
                for cs_ind, current_context_seq_len in enumerate(context_seq_len_list):
                    current_context_attn_bias = context_attn_bias[cs_ind]
                    context_attn_bias_mask = torch.ones(w.shape)
                    context_attn_bias_mask[:,:,:, current_context_start_ind:(current_context_start_ind+current_context_seq_len)] = 0
                    context_attn_bias_mask = context_attn_bias_mask.to(w.device)
                    w = w + current_context_attn_bias * (1 - context_attn_bias_mask)
                    current_context_start_ind = current_context_start_ind + current_context_seq_len

            
        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, context_seq, layer_past=None, attention_mask=None, head_mask=None, cs_self_attn_mask_prob=0, history_seq_len=None, context_attn_bias=0, context_seq_len_list=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)

        if context_seq is not None:
            context_seq_len = context_seq.shape[1]
            context_seq = self.ref_source_attn(context_seq)
            key_context_seq, value_context_seq = context_seq.split(self.split_size, dim=2)

            # Prepend keys and values with context_seq keys and values
            prepended_key = torch.cat([key_context_seq, key], dim=1)
            prepended_value = torch.cat([value_context_seq, value], dim=1)
            context_seq_present = True
        else:
            context_seq_len = 0
            prepended_key = key
            prepended_value = value
            context_seq_present = False

        query = self.split_heads(query)
        prepended_key = self.split_heads(prepended_key, k=True)
        prepended_value = self.split_heads(prepended_value)

        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        attn_outputs = self._attn(query, prepended_key, prepended_value, attention_mask, head_mask, cs_self_attn_mask_prob=cs_self_attn_mask_prob, history_seq_len=history_seq_len, context_seq_present=context_seq_present, 
                                    context_seq_len=context_seq_len, context_attn_bias=context_attn_bias, context_seq_len_list=context_seq_len_list)

        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs

        return outputs
    

