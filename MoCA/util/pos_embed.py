# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):  # changed to conform to long sequence

    grid_h = np.arange(grid_size[0], dtype=np.float32) #changed: 6 (nvar)
    grid_w = np.arange(grid_size[1], dtype=np.float32) #changed: num_patches
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):  # changed to conform to long sequence
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  #changed(H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  #changed (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb



#def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
#    """
#    grid_size: int of the grid height and width
#    return:
#    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
#    """
#    grid_h = np.arange(grid_size, dtype=np.float32)
#    grid_w = np.arange(grid_size, dtype=np.float32)
#    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
#    grid = np.stack(grid, axis=0)

 #   grid = grid.reshape([2, 1, grid_size, grid_size])
 #   pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
 #   if cls_token:
 #       pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
 #   return pos_embed


#def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
#    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
#    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
#    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

#    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
#    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position 
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

def interpolate_pos_embed(model, checkpoint_model,orig_size=(6,10),new_size=(3,10)): 
    '''
    Input: model: the class is definging for downstream
           checkpoint_model: pre-train weight
           orig_size = (old_num_time_patches,old_num_freq_patches) = (43,13)
    '''

    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed'] # 1 x 560 x 768 (1 x num_patches x E)
        embedding_size = pos_embed_checkpoint.shape[-1] # 768

        # number of special tokens (e.g. in this case num_extra_tokens = 1 for the cls token)
        num_patches = model.patch_embed.num_patches  
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches 
        
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] # old positions
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            


class SundialRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=10000, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim,
                          2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device,
                         dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

# helper function
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# some example:
# self.rotary_emb = SundialRotaryEmbedding(
#             self.head_dim, max_position_embeddings=config.max_position_embeddings)
# cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
# query_states, key_states = apply_rotary_pos_emb(
#     query_states, key_states, cos, sin, position_ids)