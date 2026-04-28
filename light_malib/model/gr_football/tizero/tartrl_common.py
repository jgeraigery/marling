# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import torch
import torch.nn as nn
import numpy as np

from light_malib.utils.logger import Logger

def check(input):
    if type(input) == np.ndarray:  
        input = np.copy(input)  # Create a writable copy of the input array  
        output = torch.from_numpy(input)  
    else:  
        output = input  
    return output 

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_fc(input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU(), nn.LayerNorm(output_size))

class FcEncoder(nn.Module):
    def __init__(self, fc_num, input_size, output_size, device="cuda"):
        super(FcEncoder, self).__init__()
        self.first_mlp = nn.Sequential(
                nn.Linear(input_size, output_size), nn.ReLU(), nn.LayerNorm(output_size)
            )
        self.mlp = nn.Sequential()
        for _ in range(fc_num - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(output_size, output_size), nn.ReLU(), nn.LayerNorm(output_size)
            ))

        self.device = device

    def forward(self, x):
        output = self.first_mlp(x)
        return self.mlp(output)
    

class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal,rnn_type='gru'):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        else:
            raise NotImplementedError(f'RNN type {rnn_type} has not been implemented.')

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def rnn_forward(self, x, h):
        if self.rnn_type == 'lstm':
            h = torch.split(h, h.shape[-1] // 2, dim=-1)
            h = (h[0].contiguous(), h[1].contiguous())
        x_, h_ = self.rnn(x, h)
        if self.rnn_type == 'lstm':
            h_ = torch.cat(h_, -1)
        return x_, h_

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn_forward(x.unsqueeze(0), (hxs * masks.view(1, -1, 1).repeat(self._recurrent_N, 1, hxs.shape[-1]).transpose(0, 1)).transpose(0, 1).contiguous())  
            #x= self.gru(x.unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            # hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                #TODO: Shouldn't the mask of the current index be used instead of the mask of start_idx?
                # Maybe masks[start_idx] is a mask without any 0s... then it would be right. Check here here more.
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn_forward(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs
