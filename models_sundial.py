import torch
import torch.nn as nn
from util.patch_embed import SundialPatchEmbedding
from typing import Optional, Tuple, List, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Cache, DynamicCache
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from util.pos_embed import RotaryEmbedding2D,apply_rotary_pos_emb_2d


class SundialMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
    
class SundialAttention(nn.Module):
    def __init__(self, 
                 max_position_embeddings=256,
                 layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = 768
        self.num_heads = 12
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = 0.1
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding2D(
            self.head_dim, max_position_embeddings=max_position_embeddings)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb_2d(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attention_mask, dropout_p=(self.attention_dropout if self.training else 0.0))

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class SundialDecoderLayer(nn.Module):
    def __init__(self,
                 layer_idx: int,
                 hidden_size=768,
                 hidden_act="silu",
                 intermediate_size=3072, ):
        super().__init__()
        self.self_attn = SundialAttention(layer_idx)

        self.ffn_layer = SundialMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.norm2 = torch.nn.LayerNorm(hidden_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn_layer(hidden_states)
        hidden_states = residual + hidden_states

        if not output_attentions:
            self_attn_weights = None

        if not use_cache:
            present_key_value = None
        return hidden_states, self_attn_weights, present_key_value
    
class SundialModel(nn.Module):
    def __init__(self,
                 img_size=[3,300],
                 patch_size=[1,5],
                 hidden_size=768,
                 intermediate_size=3072,
                 hidden_act="silu",
                 num_hidden_layers=12,):
        super().__init__()
        self.embed_layer = SundialPatchEmbedding(patch_size=patch_size[1])

        self.layers = nn.ModuleList(
            [SundialDecoderLayer(layer_idx,
                                 hidden_size=hidden_size,
                                 intermediate_size=intermediate_size,
                                 hidden_act=hidden_act)
             for layer_idx in range(num_hidden_layers)]
        )
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(
            self,
            input_ids: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        # input_ids is the input of time series, its shape is [batch_size, nvar, seq_len]

        use_cache = use_cache
        return_dict = return_dict 

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            inputs_embeds = self.embed_layer(input_ids) # [batch_size, num_patch, hidden_size]
            seq_length = inputs_embeds.shape[1]

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values)
            past_key_values_length = past_key_values.get_usable_length(
                seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            # position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            position_ids = position_ids.view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # 4d mask is passed through the layers
        attention_mask = None
        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2]

        hidden_states = self.norm(hidden_states)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache(
            ) if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )