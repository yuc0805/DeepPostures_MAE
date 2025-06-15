# Copyright 2024 Animesh Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from .utils import get_2d_sincos_pos_embed
"""
Pytorch model implementation
"""
# Stride >1 is not supported by pytorch padding "same". This hacks that.
# https://github.com/pytorch/pytorch/issues/3867
# https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/10
class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CNNModel(nn.Module):
    def __init__(self, amp_factor=1):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2dSame(
            in_channels=1,
            out_channels=32 * amp_factor,
            kernel_size=(5, 3),
            stride=(2, 1),
        )
        self.conv2 = Conv2dSame(
            in_channels=32 * amp_factor,
            out_channels=64 * amp_factor,
            kernel_size=(5, 1),
            stride=(2, 1),
        )
        self.conv3 = Conv2dSame(
            in_channels=64 * amp_factor,
            out_channels=128 * amp_factor,
            kernel_size=(5, 1),
            stride=(2, 1),
        )
        self.conv4 = Conv2dSame(
            in_channels=128 * amp_factor,
            out_channels=256 * amp_factor,
            kernel_size=(5, 1),
            stride=(2, 1),
        )
        self.conv5 = Conv2dSame(
            in_channels=256 * amp_factor,
            out_channels=256 * amp_factor,
            kernel_size=(5, 1),
            stride=(2, 1),
        )
        self.fc = nn.Linear(256 * amp_factor * 3 * 4, 256 * amp_factor)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNBiLSTMModel(nn.Module):
    def __init__(
        self, amp_factor, bi_lstm_win_size, num_classes
    ):
        super(CNNBiLSTMModel, self).__init__()
        self.cnn_model = CNNModel(amp_factor=amp_factor)
        self.hidden_size = 128
        self.bil_lstm = nn.LSTM(
            input_size=256 * amp_factor,
            hidden_size=self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )  # (batch, seq, feature)
        self.bi_lstm_win_size = bi_lstm_win_size
        self.num_classes = num_classes
        self.amp_factor = amp_factor

        # The pre-trained model is developed only for two classes and returns the flattened logits
        assert self.num_classes == 2
        self.fc_bilstm = nn.Linear(2 * self.hidden_size, 1)

    def forward(self, x):
        '''
        input: x: BS*window_size, 1, 100,3

        
        '''
        # CNN forward pass
        cnn_output = self.cnn_model(x) # BS,window_size, 512

        # Reshape for BiLSTM
        cnn_output = cnn_output.view(-1, self.bi_lstm_win_size, 256 * self.amp_factor)

        # BiLSTM forward pass
        lstm_output, _ = self.bil_lstm(cnn_output) # BS,window_size, 256

        # Concatenate both directions is not needed as pytorch automatically concat them and sends the result
        # Fully connected layer
        fc_output = self.fc_bilstm(lstm_output)

        # Reshape to get logits
        logits = fc_output.view(-1, self.bi_lstm_win_size) # Bs, Window_size
        return logits

# Leo
class AttentionInteractionModel(nn.Module):
    def __init__(self, base_model, base_model_hidden_dim=512,
                 window_size=42,num_classes=2,num_layer=1,
                 hidden_dim=256,num_heads=8,ffn_multiplier=2):
        super(AttentionInteractionModel, self).__init__()
        self.base_model = base_model
        self.window_size = window_size
        self.proj = nn.Linear(base_model_hidden_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1,window_size, hidden_dim),requires_grad=False) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*ffn_multiplier,
            batch_first=True,
            activation=F.gelu,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        if num_classes == 2:
            self.head = nn.Linear(hidden_dim, 1)    
        else:
            self.head = nn.Linear(hidden_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [1, int(self.patch_embed.num_patches)], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        

    def forward(self, x):
        '''
        input: x [BS, 42, 100, 3]
        '''
        B, W, _, _ = x.shape
        x = rearrange(x, 'b w l c -> (b w) 1 l c' )
        x = self.base_model(x) # BS*window_size,base_model_hidden_dim
        x = rearrange(x, '(b w) d -> b w d', b=B, w=W)

        x = self.proj(x)
        x = self.attn(x) # BS, 42, 256
        x = self.head(x) # BS, 42, num_classes

        return x

class MoCABiLSTMModel(nn.Module):
    def __init__(self, feature_extractor, interaction_layer,num_classes=2):
        super(MoCABiLSTMModel, self).__init__()
        '''
        feature extractor: A ViT model
        interaction_layer: A Transformer layer or a BiLSTM layer
        win_size: The window size for the sample. i.e. (BS,win_size, L, nvar)
        '''
        self.num_classes = num_classes
        self.feature_extractor_hidden_size = feature_extractor.patch_embed.proj.out_channels
        self.feature_extractor = feature_extractor
        self.interaction_layer = interaction_layer
        self.proj = nn.Linear(self.feature_extractor_hidden_size, self.interaction_layer.input_size) # 768 -> 512

        # hardcode for two classes for now
        assert self.num_classes == 2 
        self.output_proj = nn.Linear(2*self.interaction_layer.hidden_size, 1)  

    def forward(self, x):
        '''
        input: x [BS, 42, 100, 3]
        '''
       
        B, W, _, _ = x.shape
        # MoCA input needs BS*42, 1, 3, 100
        x = rearrange(x, 'b w l c -> (b w) 1 c l' )
        x = self.feature_extractor(x) # BS*window_size, hidden_size
        x = rearrange(x, '(b w) d -> b w d', b=B, w=W)
        x = self.proj(x)
        x,_ = self.interaction_layer(x) # bs, win_size, hidden_size
        x = self.output_proj(x) # bs, win_size, 1

        return x  

        