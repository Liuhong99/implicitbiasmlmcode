# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# DeepSpeed note, code taken from commit 3d59216cec89a363649b4fe3d15295ba936ced0f
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py
# Deepspeed code taken from commit: 35b4582486fe096a5c669b6ca35a3d5c6a1ec56b
# https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert
# RMS Norm taken from: https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/norms.py
#
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.nn import CrossEntropyLoss, Module
from torch.nn.modules.loss import MSELoss
from torch.nn.parameter import Parameter
from torch.utils import checkpoint
from transformers import BertConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


def get_deepspeed_config(args):
    if hasattr(args, "deepspeed_config") and args.deepspeed_config:
        from deepspeed import DeepSpeedConfig

        return DeepSpeedConfig(None, param_dict=args.ds_config)
    else:
        raise RuntimeError("deepspeed_config is not found in args.")

def merlin_linear(ftt,target1,train_num, val_num, class_num):

    target_data_test = ftt[train_num:]
    target_data_test = target_data_test / (torch.norm(target_data_test, dim = -1).reshape(val_num, 1))
    target_data_train = ftt[0:train_num]
    target_data_train = target_data_train / (torch.norm(target_data_train, dim = -1).reshape(train_num, 1))
    target_gram = torch.clamp(target_data_train.mm(target_data_train.transpose(dim0 = 1, dim1 = 0)),-0.9999999,0.9999999)
    target_kernel = target_gram
    test_gram = torch.clamp(target_data_test.mm(target_data_train.transpose(dim0 = 1, dim1 = 0)),-0.9999999,0.9999999)
    test_kernel = test_gram

    #process label
    target_label0 = target1[0:train_num]
    target_train_label = F.one_hot(target_label0, class_num) - 1 / class_num
    target_label1 = target1[train_num:]
    target_val_label = F.one_hot(target_label1, class_num) - 1 / class_num

    #ridge regression
    target_test_pred = test_kernel.mm(torch.inverse(target_kernel + 0.001 * torch.eye(train_num).to(target_kernel.device))).mm(target_train_label)
    loss_fct = MSELoss()
    loss = loss_fct(target_test_pred.view(-1), target_val_label.view(-1))
    return loss
        
@torch.jit.script
def f_gelu(x):
    return F.gelu(x)


# @torch.jit.script
# def f_gelu(x):
#     pdtype = x.dtype
#     x = x.float()
#     y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
#     return y.to(pdtype)


# @torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return F.gelu(x)


# def bias_gelu(bias, y):
#     x = bias + y
#     return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


@torch.jit.script
def bias_relu(bias, y):
    x = bias + y
    return F.relu(x)


# @torch.jit.script
# def bias_gelu(bias, y):
#     x = bias + y
#     return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


@torch.jit.script
def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return f_gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish, "tanh": F.tanh}


class LinearActivation(Module):
    r"""Fused Linear and activation Module."""
    __constants__ = ["bias"]

    def __init__(self, in_features, out_features, act="gelu", bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused_gelu = False
        self.fused_tanh = False
        self.fused_relu = False
        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, unicode)):
            if bias and act == "gelu":
                self.fused_gelu = True
            elif bias and act == "tanh":
                self.fused_tanh = True
            elif bias and act == "relu":
                self.fused_relu = True
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inp):
        if self.fused_gelu:
            return bias_gelu(self.bias, F.linear(inp, self.weight, None))
        elif self.fused_tanh:
            return bias_tanh(self.bias, F.linear(inp, self.weight, None))
        elif self.fused_relu:
            return bias_relu(self.bias, F.linear(inp, self.weight, None))
        else:
            return self.act_fn(F.linear(inp, self.weight, self.bias))

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class RegularLinearActivation(Module):
    """Regular Linear activation module with"""

    def __init__(self, in_features, out_features, act="gelu"):
        super(RegularLinearActivation, self).__init__()
        self.dense = nn.Linear(in_features, out_features)
        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, unicode)):
            self.act = ACT2FN[act]

    def forward(self, hidden_states):
        return self.act(self.dense(hidden_states))


def get_apex_layer_norm():
    try:
        import apex

        # apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
        import apex.normalization

        # apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
        return apex.normalization.FusedLayerNorm
    except ImportError:
        raise Exception(f"Layer norm of type apex is not available, apex not installed.")


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias

        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(dim))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


LAYER_NORM_TYPES = {"pytorch": nn.LayerNorm, "apex": get_apex_layer_norm(), "rms_norm": RMSNorm}


def get_layer_norm_type(config):
    if config.layer_norm_type in LAYER_NORM_TYPES:
        return LAYER_NORM_TYPES[config.layer_norm_type]
    else:
        raise Exception(f"Layer norm of type {config.layer_norm_type} is not available.")


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        BertLayerNorm = get_layer_norm_type(config)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if self.config.useLN:
            embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        context_layer, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(context_layer, input_tensor)
        output = (
            attention_output,
            attention_probs,
        )
        return output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        if config.fused_linear_layer:
            linear_layer = LinearActivation
        else:
            linear_layer = RegularLinearActivation
        self.dense_act = linear_layer(
            config.hidden_size, config.intermediate_size, act=config.hidden_act
        )

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.config = config

        BertLayerNorm = get_layer_norm_type(config)

        self.PreAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.PostAttentionLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def maybe_layer_norm(self, hidden_states, layer_norm, current_ln_mode):
        if self.config.useLN and self.config.encoder_ln_mode in current_ln_mode:
            return layer_norm(hidden_states)
        else:
            return hidden_states

    def forward(self, hidden_states, attention_mask, action=1, keep_prob=1.0):
        attention_probs = None
        intermediate_input = None

        if action == 0:
            intermediate_input = hidden_states
        else:
            pre_attn_input = self.maybe_layer_norm(
                hidden_states, self.PreAttentionLayerNorm, "pre-ln"
            )
            self_attn_out = self.attention(pre_attn_input, attention_mask)

            attention_output, attention_probs = self_attn_out
            attention_output = attention_output * 1 / keep_prob

            intermediate_input = hidden_states + attention_output
            intermediate_input = self.maybe_layer_norm(
                intermediate_input, self.PreAttentionLayerNorm, "post-ln"
            )

        if action == 0:
            layer_output = intermediate_input
        else:
            intermediate_pre_ffn = self.maybe_layer_norm(
                intermediate_input, self.PostAttentionLayerNorm, "pre-ln"
            )
            intermediate_output = self.intermediate(intermediate_pre_ffn)

            layer_output = self.output(intermediate_output)
            layer_output = layer_output * 1 / keep_prob

            layer_output = layer_output + intermediate_input
            layer_output = self.maybe_layer_norm(
                layer_output, self.PostAttentionLayerNorm, "post-ln"
            )

        output = (
            layer_output,
            attention_probs,
        )
        return output


class BertEncoder(nn.Module):
    def __init__(self, config, args):
        super(BertEncoder, self).__init__()
        self.config = config
        BertLayerNorm = get_layer_norm_type(config)
        self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.is_transformer_kernel = (
            hasattr(args, "deepspeed_transformer_kernel") and args.deepspeed_transformer_kernel
        )

        if hasattr(args, "deepspeed_transformer_kernel") and args.deepspeed_transformer_kernel:
            from deepspeed import DeepSpeedTransformerConfig, DeepSpeedTransformerLayer

            ds_config = get_deepspeed_config(args)
            has_huggingface = hasattr(args, "huggingface")
            ds_transformer_config = DeepSpeedTransformerConfig(
                batch_size=ds_config.train_micro_batch_size_per_gpu,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                heads=config.num_attention_heads,
                attn_dropout_ratio=config.attention_probs_dropout_prob,
                hidden_dropout_ratio=config.hidden_dropout_prob,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                local_rank=args.local_rank if hasattr(args, "local_rank") else -1,
                seed=args.seed,
                fp16=ds_config.fp16_enabled,
                pre_layer_norm=True if "pre-ln" in config.encoder_ln_mode else False,
                normalize_invertible=args.normalize_invertible,
                gelu_checkpoint=args.gelu_checkpoint,
                adjust_init_range=True,
                attn_dropout_checkpoint=args.attention_dropout_checkpoint,
                stochastic_mode=args.stochastic_mode,
                huggingface=has_huggingface,
                training=self.training,
            )

            self.layer = nn.ModuleList(
                [
                    copy.deepcopy(DeepSpeedTransformerLayer(ds_transformer_config))
                    for _ in range(config.num_hidden_layers)
                ]
            )
        else:
            layer = BertLayer(config)
            self.layer = nn.ModuleList(
                [copy.deepcopy(layer) for _ in range(self.config.num_hidden_layers)]
            )

    def add_attention(self, all_attentions, attention_probs):
        if attention_probs is not None:
            all_attentions.append(attention_probs)

        return all_attentions

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
        output_attentions=False,
    ):
        all_encoder_layers = []
        all_attentions = []

        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(
                    custom(l, l + chunk_length), hidden_states, attention_mask * 1
                )
                l += chunk_length
            # decoder layers
        else:
            for layer_module in self.layer:
                if self.is_transformer_kernel:
                    # using Deepspeed Transformer kernel
                    hidden_states = layer_module(hidden_states, attention_mask)
                else:
                    layer_out = layer_module(
                        hidden_states,
                        attention_mask,
                    )
                    hidden_states, attention_probs = layer_out
                    # get all attention_probs from layers
                    if output_attentions:
                        all_attentions = self.add_attention(all_attentions, attention_probs)

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers or checkpoint_activations:
            if self.config.useLN and self.config.encoder_ln_mode in "pre-ln":
                hidden_states = self.FinalLayerNorm(hidden_states)

            all_encoder_layers.append(hidden_states)
        outputs = (all_encoder_layers,)
        if output_attentions:
            outputs += (all_attentions,)
        return outputs


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        if config.fused_linear_layer:
            linear_layer = LinearActivation
        else:
            linear_layer = RegularLinearActivation
        self.dense_act = linear_layer(config.hidden_size, config.hidden_size, act="tanh")

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output

class BertPoolerCYK(nn.Module):
    def __init__(self, config):
        super(BertPoolerCYK, self).__init__()
        if config.fused_linear_layer:
            linear_layer = LinearActivation
        else:
            linear_layer = RegularLinearActivation
        self.dense_act = linear_layer(config.hidden_size, config.hidden_size, act="tanh")

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output
    
class BertPoolerL(nn.Module):
    def __init__(self, config):
        super(BertPoolerL, self).__init__()
        if config.fused_linear_layer:
            linear_layer = LinearActivation
        else:
            linear_layer = RegularLinearActivation
        self.dense_act = linear_layer(config.hidden_size * 126, config.hidden_size * 126, act="tanh")

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = torch.flatten(hidden_states[:, 1:-1, :], start_dim = 1)
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output

class BertPoolerC(nn.Module):
    def __init__(self, config):
        super(BertPoolerC, self).__init__()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = torch.flatten(hidden_states[:, 1:-1, :], start_dim = 1)
        
        return first_token_tensor

class BertPoolerCM(nn.Module):
    def __init__(self, config):
        super(BertPoolerCM, self).__init__()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = torch.mean(hidden_states[:, 1:-1, :], dim = 1)
        
        return first_token_tensor
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.config = config
        if config.fused_linear_layer:
            linear_layer = LinearActivation
        else:
            linear_layer = RegularLinearActivation
        self.dense_act = linear_layer(config.hidden_size, config.hidden_size, act=config.hidden_act)
        BertLayerNorm = get_layer_norm_type(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        if self.config.useLN:
            hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        self.sparse_predict = config.sparse_mask_prediction
        if not config.sparse_mask_prediction:
            self.decoder.bias = self.bias

    def forward(self, hidden_states, masked_token_indexes):
        if self.sparse_predict:
            if masked_token_indexes is not None:
                hidden_states = hidden_states.contiguous().view(-1, hidden_states.shape[-1])[
                    masked_token_indexes
                ]
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        if not self.sparse_predict:
            hidden_states = torch.index_select(
                hidden_states.contiguous().view(-1, hidden_states.shape[-1]), 0, masked_token_indexes
            )
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output, masked_token_indexes=None):
        prediction_scores = self.predictions(sequence_output, masked_token_indexes)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_token_indexes=None):
        prediction_scores = self.predictions(sequence_output, masked_token_indexes)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, inputs=inputs, kwargs=kwargs)
        print(inputs)
        print(kwargs)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args=None):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = (
            config.pad_token_id
            if hasattr(config, "pad_token_id") and config.pad_token_id is not None
            else 0
        )
        self.encoder = BertEncoder(config, args)
        self.pooler = BertPooler(config)

        logger.info("Init BERT pretrain model")

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
        output_attentions=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = extended_attention_mask.to(
            dtype=self.embeddings.word_embeddings.weight.dtype  # should be of same dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoder_output = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations,
            output_attentions=output_attentions,
        )
        encoded_layers = encoder_output[0]
        sequence_output = encoded_layers[-1]

        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        output = (
            encoded_layers,
            pooled_output,
        )
        if output_attentions:
            output += (encoder_output[-1],)
        return output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config, args)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, batch):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[5]
        next_sentence_label = batch[4]
        checkpoint_activations = False

        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        if masked_lm_labels is not None and next_sentence_label is not None:
            # filter out all masked labels.
            masked_token_indexes = torch.nonzero(
                (masked_lm_labels + 1).view(-1), as_tuple=False
            ).view(-1)
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_token_indexes
            )
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
            return prediction_scores, seq_relationship_score


class BertLMHeadModel(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertMultiHead(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config, args):
        super(BertLMHeadModel, self).__init__(config)
        self.bert = BertModel(config, args)

        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.init_weights()

    def forward(self, batch, output_attentions=False):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[4]
        checkpoint_activations = False

        bert_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        sequence_output = bert_output[0]

        if masked_lm_labels is None:
            prediction_scores = self.cls(sequence_output)
            return prediction_scores

        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(
            -1
        )
        prediction_scores = self.cls(sequence_output, masked_token_indexes)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)

            outputs = (masked_lm_loss,)
            if output_attentions:
                outputs += (bert_output[-1],)
            return outputs
        else:
            return prediction_scores, sequence_output


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        next_sentence_label=None,
        checkpoint_activations=False,
    ):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args=None):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        pooled_output = self.pooler(outputs[0])
        #print(pooled_output.shape)
        
        

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
class BertForSequenceClassificationMerlin(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(BertForSequenceClassificationMerlin, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        pooled_output = self.pooler(outputs[0])
        #print(pooled_output.shape)
        
        
        
        
        
        

        pooled_output = self.dropout(pooled_output)
        
        train_num = pooled_output.shape[0] // 8 * 7
        val_num = pooled_output.shape[0] - train_num
        class_num = 2
        target1 = labels
        
        loss_merlin = merlin_linear(pooled_output,target1,train_num, val_num, class_num)
        
        
        
        logits = self.classifier(pooled_output.detach())

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        ),loss_merlin
    
class BertForSequenceClassificationL(BertPreTrainedModel):
    

    def __init__(self, config, args=None):
        super(BertForSequenceClassificationL, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)

        self.classifier = nn.Linear(config.hidden_size * 18, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        
        #print(outputs[0].shape)
        pooled_output = self.pooler(outputs[0])
        #print(pooled_output.shape)
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class BertForSequenceClassificationLOPT(BertPreTrainedModel):
    

    def __init__(self, config, args=None):
        super(BertForSequenceClassificationLOPT, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerCM(config)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        
        #print(outputs[0].shape)
        pooled_output = self.pooler(outputs[0])
        #print(pooled_output.shape)
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
class BertForSequenceClassificationLA(BertPreTrainedModel):
    

    def __init__(self, config, args=None):
        super(BertForSequenceClassificationLA, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        

        self.classifier = nn.Linear(256 * 8, self.num_labels)
        self.attn = PartitionedMultiHeadAttention(config.hidden_size, 4, 64)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        attn_output = self.attn(outputs[0])
        attn_output = torch.flatten(attn_output[:,:,1:-1,:], start_dim = 1)
        
        #print(pooled_output.shape)
        
        
  
        pooled_output = self.dropout(attn_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
class BertForSequenceClassificationCond(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForSequenceClassificationCond, self).__init__(config)
        self.bert = BertModel(config, args)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)
        self.classifier = nn.Linear(config.vocab_size * 18, self.num_labels)

        self.init_weights()

    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,):

        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        sequence_output = outputs[0]
        prediction_scores = F.softmax(self.cls(sequence_output), dim = -1)

        pooled_output = self.pooler(prediction_scores)
        #print(pooled_output.shape)
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class BertForSequenceClassificationCondF(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForSequenceClassificationCondF, self).__init__(config)
        self.bert = BertModel(config, args)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)
        self.classifier = nn.Linear(config.vocab_size * 18, self.num_labels)

        self.init_weights()

    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,):
        sequence_output_list = []
        for i in range(20):
            masked_input_ids = copy.deepcopy(input_ids)
            masked_input_ids[:,i] = 103
            outputs = self.bert(
            masked_input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
            )
            sequence_output_list.append(outputs[0][:,i,:].unsqueeze(1))
        sequence_output = torch.cat(sequence_output_list, dim = 1)

        prediction_scores = F.softmax(self.cls(sequence_output), dim = -1)

        pooled_output = self.pooler(prediction_scores)
        
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
   
class BertForSequenceClassificationCondFS(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForSequenceClassificationCondFS, self).__init__(config)
        self.bert = BertModel(config, args)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)
        self.classifier = nn.Linear(config.vocab_size * 8, self.num_labels)

        self.init_weights()

    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,):
        sequence_output_list = []
        for i in range(10):
            masked_input_ids = copy.deepcopy(input_ids)
            masked_input_ids[:,i] = 103
            outputs = self.bert(
            masked_input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
            )
            sequence_output_list.append(outputs[0][:,i,:].unsqueeze(1))
        sequence_output = torch.cat(sequence_output_list, dim = 1)

        prediction_scores = self.cls(sequence_output)

        pooled_output = self.pooler(prediction_scores)
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class PartitionedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_model, n_head, d_qkv, attention_dropout=0.1, initializer_range=0.02
    ):
        super().__init__()

        self.w_qkv_c = nn.Parameter(torch.Tensor(n_head, d_model, 3, d_qkv))
        
        self.w_o_c = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))

        bound = math.sqrt(3.0) * initializer_range
        for param in [self.w_qkv_c, self.w_o_c]:
            nn.init.uniform_(param, -bound, bound)
        self.scaling_factor = 1 / d_qkv

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, mask=None):

        qkv_c = torch.einsum("btf,hfca->bhtca", x, self.w_qkv_c)
        q, k, v = [c.squeeze(dim=3) for c in torch.chunk(qkv_c, 3, dim=3)]
        q = q * self.scaling_factor
        
        dots = torch.einsum("bhqa,bhka->bhqk", q, k)
        if mask is not None:
            dots.data.masked_fill_(~mask[:, None, None, :], -float("inf"))
        probs = F.softmax(dots, dim=-1)
        probs = self.dropout(probs)
        o = torch.einsum("bhqk,bhka->bhqa", probs, v)
        return o
    
class BertForSequenceClassificationCondFA(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertForSequenceClassificationCondFA, self).__init__(config)
        self.bert = BertModel(config, args)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)
        self.classifier = nn.Linear(16384, self.num_labels)
        self.attn = PartitionedMultiHeadAttention(config.vocab_size, 4, 64)

        self.init_weights()

    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,):
        sequence_output_list = []
        for i in range(10):
            masked_input_ids = copy.deepcopy(input_ids)
            masked_input_ids[:,i] = 103
            outputs = self.bert(
            masked_input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
            )
            sequence_output_list.append(outputs[0][:,i,:].unsqueeze(1))
        sequence_output = torch.cat(sequence_output_list, dim = 1)

        prediction_scores = F.softmax(self.cls(sequence_output), dim = -1)
        prediction_scores1 = self.attn(prediction_scores)
        #print(prediction_scores1.shape)
        

        pooled_output = self.pooler(prediction_scores1)
        
        
        
  
        pooled_output = self.dropout(pooled_output)
        #print(pooled_output.shape)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    



class RNNLayer(nn.Module):
    def __init__(self, config):
        super(RNNLayer, self).__init__()
        self.config = config
        
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        RNNLayerNorm = get_layer_norm_type(config)

        self.PreAttentionLayerNorm = RNNLayerNorm(config.hidden_size, eps=1e-12)
        self.PostAttentionLayerNorm =  RNNLayerNorm(config.hidden_size, eps=1e-12)
        #self.intermediate = BertIntermediate(config)
        #self.output = BertOutput(config)

    def maybe_layer_norm(self, hidden_states, layer_norm, current_ln_mode):
        if self.config.useLN and self.config.encoder_ln_mode in current_ln_mode:
            return layer_norm(hidden_states)
        else:
            return hidden_states

    def forward(self, hidden_states, attention_mask, action=1, keep_prob=1.0):
        intermediate_input = None

        if action == 0:
            intermediate_input = hidden_states
        else:
            pre_attn_input = self.maybe_layer_norm(
                hidden_states, self.PreAttentionLayerNorm, "pre-ln"
            )
            self_attn_out,_ = self.lstm(pre_attn_input)

            attention_output = self_attn_out
            attention_output = attention_output * 1 / keep_prob

            intermediate_input = hidden_states + attention_output
            intermediate_input = self.maybe_layer_norm(
                intermediate_input, self.PreAttentionLayerNorm, "post-ln"
            )

        if action == 0:
            layer_output = intermediate_input
        else:
            intermediate_pre_ffn = self.maybe_layer_norm(
                intermediate_input, self.PostAttentionLayerNorm, "pre-ln"
            )
            #intermediate_output = self.intermediate(intermediate_pre_ffn)
            layer_output = intermediate_pre_ffn
            #layer_output = self.output(intermediate_output)
            layer_output = layer_output * 1 / keep_prob

            #layer_output = layer_output + intermediate_input
            layer_output = self.maybe_layer_norm(
                layer_output, self.PostAttentionLayerNorm, "post-ln"
            )

        output = layer_output
        return output

class RNNEncoder(nn.Module):
    def __init__(self, config, args):
        super(RNNEncoder, self).__init__()
        self.config = config
        RNNLayerNorm = get_layer_norm_type(config)
        self.FinalLayerNorm = RNNLayerNorm(config.hidden_size, eps=1e-12)
        
        layer = RNNLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(self.config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
        output_attentions=False,
    ):
        all_encoder_layers = []

        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        if checkpoint_activations:
            l = 0
            num_layers = len(self.layer)
            chunk_length = math.ceil(math.sqrt(num_layers))
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(
                    custom(l, l + chunk_length), hidden_states, attention_mask * 1
                )
                l += chunk_length
            # decoder layers
        else:
            for layer_module in self.layer:
                
                layer_out = layer_module(
                    hidden_states,
                    attention_mask,
                )
                hidden_states = layer_out
                    
                    

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers or checkpoint_activations:
            if self.config.useLN and self.config.encoder_ln_mode in "pre-ln":
                hidden_states = self.FinalLayerNorm(hidden_states)

            all_encoder_layers.append(hidden_states)
        outputs = (all_encoder_layers,)
        return outputs

class RNNModel(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super(RNNModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = (
            config.pad_token_id
            if hasattr(config, "pad_token_id") and config.pad_token_id is not None
            else 0
        )
        self.encoder = RNNEncoder(config, args)
        self.pooler = BertPooler(config)

        logger.info("Init RNN pretrain model")

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = extended_attention_mask.to(
            dtype=self.embeddings.word_embeddings.weight.dtype  # should be of same dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoder_output = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations,
        )
        encoded_layers = encoder_output[0]
        sequence_output = encoded_layers[-1]

        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        output = (
            encoded_layers,
            pooled_output,
        )

        return output


class RNNForPreTraining(BertPreTrainedModel):

    def __init__(self, config, args):
        super(RNNForPreTraining, self).__init__(config)
        self.rnn = RNNModel(config, args)
        self.cls = BertPreTrainingHeads(config, self.rnn.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, batch):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[5]
        next_sentence_label = batch[4]
        checkpoint_activations = False

        sequence_output, pooled_output = self.rnn(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        if masked_lm_labels is not None and next_sentence_label is not None:
            # filter out all masked labels.
            masked_token_indexes = torch.nonzero(
                (masked_lm_labels + 1).view(-1), as_tuple=False
            ).view(-1)
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_token_indexes
            )
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
            return prediction_scores, seq_relationship_score


class RNNLMHeadModel(BertPreTrainedModel):
    
    def __init__(self, config, args):
        super(RNNLMHeadModel, self).__init__(config)
        self.rnn = RNNModel(config, args)

        self.cls = BertOnlyMLMHead(config, self.rnn.embeddings.word_embeddings.weight)
        self.init_weights()

    def forward(self, batch, output_attentions=False):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[4]
        checkpoint_activations = False

        bert_output = self.rnn(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        sequence_output = bert_output[0]

        if masked_lm_labels is None:
            prediction_scores = self.cls(sequence_output)
            return prediction_scores

        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(
            -1
        )
        prediction_scores = self.cls(sequence_output, masked_token_indexes)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)

            outputs = (masked_lm_loss,)
            return outputs
        else:
            return prediction_scores

class RNNForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, args=None):
        super(RNNForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.rnn = RNNModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.rnn(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        pooled_output = self.pooler(outputs[0])
        #print(pooled_output.shape)
        
        

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
class RNNForSequenceClassificationL(BertPreTrainedModel):
    

    def __init__(self, config, args=None):
        super(RNNForSequenceClassificationL, self).__init__(config)
        self.num_labels = config.num_labels
        self.rnn = RNNModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)

        self.classifier = nn.Linear(config.hidden_size * 18, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.rnn(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        
        #print(outputs[0].shape)
        pooled_output = self.pooler(outputs[0])
        #print(pooled_output.shape)
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )    

    
class RNNForSequenceClassificationCondF(BertPreTrainedModel):
    def __init__(self, config, args):
        super(RNNForSequenceClassificationCondF, self).__init__(config)
        self.rnn = RNNModel(config, args)
        self.cls = BertOnlyMLMHead(config, self.rnn.embeddings.word_embeddings.weight)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)
        self.classifier = nn.Linear(config.vocab_size * 18, self.num_labels)

        self.init_weights()

    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,):
        sequence_output_list = []
        for i in range(20):
            masked_input_ids = copy.deepcopy(input_ids)
            masked_input_ids[:,i] = 103
            outputs = self.rnn(
            masked_input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
            )
            sequence_output_list.append(outputs[0][:,i,:].unsqueeze(1))
        sequence_output = torch.cat(sequence_output_list, dim = 1)

        prediction_scores = F.softmax(self.cls(sequence_output), dim = -1)

        pooled_output = self.pooler(prediction_scores)
        
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
import math, time
from torch.autograd import Variable
import random

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    
class ConcatPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=10000):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
        nn.init.normal_(self.timing_table)
        self.norm = nn.LayerNorm(d_model * 2)

    def forward(self, x):
        timing = self.timing_table[: x.shape[1], :].unsqueeze(0)
        x, timing = torch.broadcast_tensors(x, timing)
        out = torch.cat([x, timing], dim=-1)
        out = self.norm(out)
        return out
    
c = copy.deepcopy
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
'''
class Max_tensor(nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_features: int,
                 device=None, dtype=None) -> None:
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Max_tensor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = nn.Linear(in_features, mid_features)
        self.ln1 = torch.nn.LayerNorm(mid_features)
        self.relu1 = nn.GELU()
        self.linear2 = nn.Linear(mid_features, mid_features)
        self.ln2 = torch.nn.LayerNorm(mid_features)
        self.relu2 = nn.GELU()
        self.linear4 = nn.Linear(mid_features, out_features)
        
        #self.reset_parameters()


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        xx = input
        xx = xx + self.relu1(self.ln1(self.linear1(xx)))
        xx = xx + self.relu2(self.ln2(self.linear2(xx)))
        xx = self.linear4(xx)
        
        return xx

class Max_L(nn.Module):
    def __init__(self, temperature: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Max_L, self).__init__()
        if temperature:
            self.temperature = nn.Parameter(torch.empty(1, **factory_kwargs))
        else:
            self.temperature = 1
        self.reset_parameters()
        self.softmax = nn.Softmax(dim = -2)

    def reset_parameters(self) -> None:
        if not isinstance(self.temperature, int):
            nn.init.ones_(self.temperature)


    def forward(self, input: torch.Tensor, input1: torch.Tensor) -> torch.Tensor:
        mask = self.softmax(input * self.temperature)
        mask = mask+ torch.normal(0,0.01,mask.shape).cuda()
        
        return (torch.sum(input * mask, dim = -2), torch.sum(input1 * mask, dim = -2))

    def extra_repr(self) -> str:
        return 'temperature={}'.format(
            self.temperature is not None
        )

class CYKEncoder(nn.Module):
    def __init__(self, config, args) -> None:
        super(CYKEncoder, self).__init__()
        
        self.d_feat = config.hidden_size
        dropout = 0.1

        c = copy.deepcopy
        ff = PositionwiseFeedForward(3*config.hidden_size, 3*config.intermediate_size, dropout)
        attn = MultiHeadedAttention(config.num_attention_heads, 3*config.hidden_size)
        
        
        self.max_prob = Max_tensor(2*config.hidden_size, config.hidden_size, config.intermediate_size)
        self.max_pos = Max_tensor(2*config.hidden_size, config.hidden_size, config.intermediate_size)
        
        self.attn = Encoder(EncoderLayer(3*config.hidden_size, c(attn), c(ff), dropout), config.num_hidden_layers)
        self.pe = ConcatPositionalEncoding(config.hidden_size)
        self.linear = nn.Linear(36 * 3*self.d_feat, self.d_feat)
        self.max_l = Max_L()

    def shift(self, x, dist, lr):
        B, l, n = x.shape
        new_vec = torch.zeros((B, dist, n)).cuda()#.to_device(x.device)
        if lr == 'l':
            return torch.cat([x, new_vec], dim = 1)
        else:
            return torch.cat([new_vec, x], dim = 1)


    def forward(self, x):
        B = x.shape[0]
        T = x.shape[1]
        v_list = []
        w_list = []
        v_s_list = []
        v_list.append(x)
        w_list.append(x)
        v_s_list.append([self.shift(x, 1, 'l'),self.shift(x, 1, 'r')])
        for t in range(1,T):
            tp_list = []
            r_list = []
            for i in range(t):
                tp_list.append(torch.cat([v_s_list[i][1], v_s_list[t - 1 - i][0]], dim = -1)[:,i+1:T+i-t+1,:].unsqueeze(-2))
            tp_prob = torch.cat(tp_list, dim = -2)
            u = self.max_prob(tp_prob)
            r = self.max_pos(tp_prob)

            curr_v, curr_w = self.max_l(u, r)
            v_list.append(curr_v)
            w_list.append(curr_w)
            v_s_list.append([self.shift(curr_v,t+1, 'l'),self.shift(curr_v, t+1, 'r')])
        

        w_list_square = []
        for i in range(T):
            w_list_square.append(w_list[T-1-i])
        for i in range(1,T):
            w_list_square.append(w_list[i])
            
        v_list_square = []
        for i in range(T):
            v_list_square.append(v_list[T-1-i])
        for i in range(1,T):
            v_list_square.append(v_list[i])
        
        
        w_feat = torch.cat(w_list_square, dim = 1)
        w_feat = self.pe(w_feat)
        v_feat = torch.cat(v_list_square, dim = 1)
        #v_feat = self.pe(v_feat)
        
        wv_feat = torch.cat([v_feat,w_feat], dim = -1)
        
        #for j in range(self.max_attn):
        wv_feat = self.attn(wv_feat, None)
            
        wv_feat = torch.flatten(wv_feat.reshape(B,T,T, 3*self.d_feat), start_dim = 2)
        wv_feat = self.linear(wv_feat)
        
        
        return wv_feat
'''
class Max_tensor(nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_features: int,
                 device=None, dtype=None) -> None:
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Max_tensor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = nn.Linear(in_features, mid_features)
        self.relu1 = nn.GELU()
        self.linear2 = nn.Linear(mid_features, out_features)
        self.ln = torch.nn.LayerNorm(out_features)
        
        #self.reset_parameters()


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        xx = input
        xx = xx + self.ln(self.linear2(self.relu1(self.linear1(xx))))
        
        return xx

class Max_L(nn.Module):
    def __init__(self, temperature: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Max_L, self).__init__()


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sum(input, dim = -2)

    def extra_repr(self) -> str:
        return 'temperature={}'.format(
            self.temperature is not None
        )
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self,t):
        x = self.pe[:, t]
        return self.dropout(x)
    
class CYKEncoder(nn.Module):
    def __init__(self, config, args) -> None:
        super(CYKEncoder, self).__init__()
        
        self.d_feat = config.hidden_size
        self.max_prob = Max_tensor(config.hidden_size, config.hidden_size, config.intermediate_size)
        self.max_l = Max_L()
        self.refmat = torch.zeros(20,1,20,self.d_feat)
        self.pe = PositionalEncoding(self.d_feat, 0.1)
        
        for t in range(20):
            self.refmat[t,:,t,:] = self.pe(t)
        print(self.refmat.shape)

    def shift(self, x, T, dist, lr):
        _,B, l, n = x.shape
        new_vec = x.new_zeros((T,B, dist, n))#.to_device(x.device)
        #print(x.shape, new_vec.shape)
        if lr == 'l':
            return torch.cat([x, new_vec], dim = 2)
        else:
            return torch.cat([new_vec, x], dim = 2)
                             
    def shifta(self, x, dist, lr):
        B, l, n = x.shape
        new_vec = x.new_zeros((B, dist, n))#.to_device(x.device)
        #print(x.shape, new_vec.shape)
        if lr == 'l':
            return torch.cat([x, new_vec], dim = 1)
        else:
            return torch.cat([new_vec, x], dim = 1)


    def forward(self, x):
        B = x.shape[0]
        T = x.shape[1]
        x = x.repeat(T,1,1).reshape(T,B,T,self.d_feat) + self.refmat.to(x.device).type(x.type())
        v_list = []
        v_s_list = []
        v_list.append(x)
        v_s_list.append([self.shift(x, T, 1, 'l'),self.shift(x, T, 1, 'r')])
        for t in range(1,T):
            tp_list = []
            r_list = []
            for i in range(t):
                tp_list.append((v_s_list[i][1] + v_s_list[t - 1 - i][0])[:,:,i+1:T+i-t+1,:].unsqueeze(-2))
            tp_prob = torch.cat(tp_list, dim = -2)
            #print(tp_prob.shape)
            u = self.max_prob(tp_prob)

            curr_v = self.max_l(u)
            v_list.append(curr_v)
            v_s_list.append([self.shift(curr_v,T,t+1, 'l'),self.shift(curr_v,T, t+1, 'r')])
        

        v_feat = curr_v.squeeze(-2).transpose(0,1)
        return v_feat
    
    def forwarda(self, x):
        B = x.shape[0]
        T = x.shape[1]
        v_list = []
        v_s_list = []
        v_list.append(x)
        v_s_list.append([self.shifta(x, 1, 'l'),self.shifta(x, 1, 'r')])
        for t in range(1,T):
            tp_list = []
            r_list = []
            for i in range(t):
                tp_list.append((v_s_list[i][1] + v_s_list[t - 1 - i][0])[:,i+1:T+i-t+1,:].unsqueeze(-2))
            tp_prob = torch.cat(tp_list, dim = -2)
            #print(tp_prob.shape)
            u = self.max_prob(tp_prob)

            curr_v = self.max_l(u)
            v_list.append(curr_v)
            v_s_list.append([self.shifta(curr_v,t+1, 'l'),self.shifta(curr_v, t+1, 'r')])
        

        v_feat = curr_v.squeeze(-2)
        return v_feat
    
class CYKModel(BertPreTrainedModel):

    def __init__(self, config, args=None):
        super(CYKModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = (
            config.pad_token_id
            if hasattr(config, "pad_token_id") and config.pad_token_id is not None
            else 0
        )
        self.encoder = CYKEncoder(config, args)
        self.pooler = BertPooler(config)

        logger.info("Init CYK pretrain model")

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = extended_attention_mask.to(
            dtype=self.embeddings.word_embeddings.weight.dtype  # should be of same dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoder_output = self.encoder(
            embedding_output
        )
        sequence_output = encoder_output

        pooled_output = self.pooler(sequence_output)

        output = (
            sequence_output,
            pooled_output,
        )

        return output

class CYKLMHeadModel(BertPreTrainedModel):
    
    def __init__(self, config, args):
        super(CYKLMHeadModel, self).__init__(config)
        self.cyk = CYKModel(config, args)

        self.cls = BertOnlyMLMHead(config, self.cyk.embeddings.word_embeddings.weight)
        self.init_weights()

    def forward(self, batch, output_attentions=False):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[4]
        checkpoint_activations = False

        bert_output = self.cyk(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        sequence_output = bert_output[0]

        if masked_lm_labels is None:
            prediction_scores = self.cls(sequence_output)
            return prediction_scores

        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(
            -1
        )
        prediction_scores = self.cls(sequence_output, masked_token_indexes)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)

            outputs = (masked_lm_loss,)
            return outputs
        else:
            return prediction_scores
        
        
class CYKForSequenceClassification(BertPreTrainedModel):
    
    def __init__(self, config, args=None):
        super(CYKForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.cyk = CYKModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerCYK(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):
        embedding_output = self.cyk.embeddings(input_ids, token_type_ids)

        encoder_output = self.cyk.encoder.forwarda(
            embedding_output
        )
        
        
        pooled_output = self.dropout(self.pooler(encoder_output))
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
class CYKForSequenceClassificationA(BertPreTrainedModel):
    
    def __init__(self, config, args=None):
        super(CYKForSequenceClassificationA, self).__init__(config)
        self.num_labels = config.num_labels
        self.cyk = CYKModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):
        outputs = self.cyk(
            input_ids,
            token_type_ids,
            attention_mask,
        )

        pooled_output = self.pooler(outputs[0])
        #print(pooled_output.shape)
        
        

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
class CYKForSequenceClassificationL(BertPreTrainedModel):
    

    def __init__(self, config, args=None):
        super(CYKForSequenceClassificationL, self).__init__(config)
        self.num_labels = config.num_labels
        self.cyk = CYKModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)

        self.classifier = nn.Linear(config.hidden_size * 18, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.cyk(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        pooled_output = self.pooler(outputs[0])
        #print(pooled_output.shape)
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
class CYKForSequenceClassificationCondF(BertPreTrainedModel):
    def __init__(self, config, args):
        super(CYKForSequenceClassificationCondF, self).__init__(config)
        self.cyk = CYKModel(config, args)
        self.cls = BertOnlyMLMHead(config, self.cyk.embeddings.word_embeddings.weight)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)
        self.classifier = nn.Linear(config.vocab_size * 18, self.num_labels)

        self.init_weights()

    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,):
        sequence_output_list = []
        for i in range(20):
            masked_input_ids = copy.deepcopy(input_ids)
            masked_input_ids[:,i] = 103
            outputs = self.cyk(
            masked_input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
            )
            sequence_output_list.append(outputs[0][:,i,:].unsqueeze(1))
        sequence_output = torch.cat(sequence_output_list, dim = 1)

        prediction_scores = F.softmax(self.cls(sequence_output), dim = -1)

        pooled_output = self.pooler(prediction_scores)
        
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    
class CYKNEncoder(nn.Module):
    def __init__(self, config, args) -> None:
        super(CYKNEncoder, self).__init__()
        
        self.d_feat = config.hidden_size
        self.max_prob = Max_tensor(config.hidden_size, config.hidden_size, config.intermediate_size)
        self.max_l = Max_L()


    def shifta(self, x, dist, lr):
        B, l, n = x.shape
        new_vec = x.new_zeros((B, dist, n))#.to_device(x.device)
        #print(x.shape, new_vec.shape)
        if lr == 'l':
            return torch.cat([x, new_vec], dim = 1)
        else:
            return torch.cat([new_vec, x], dim = 1)


    
    def forward(self, x):
        B = x.shape[0]
        T = x.shape[1]
        v_list = []
        v_s_list = []
        v_list.append(x)
        v_s_list.append([self.shifta(x, 1, 'l'),self.shifta(x, 1, 'r')])
        for t in range(1,T):
            tp_list = []
            r_list = []
            for i in range(t):
                tp_list.append((v_s_list[i][1] + v_s_list[t - 1 - i][0])[:,i+1:T+i-t+1,:].unsqueeze(-2))
            tp_prob = torch.cat(tp_list, dim = -2)
            #print(tp_prob.shape)
            u = self.max_prob(tp_prob)

            curr_v = self.max_l(u)
            v_list.append(curr_v)
            v_s_list.append([self.shifta(curr_v,t+1, 'l'),self.shifta(curr_v, t+1, 'r')])
        

        v_feat = curr_v.squeeze(-2)
        return v_feat
    
class CYKNModel(BertPreTrainedModel):

    def __init__(self, config, args=None):
        super(CYKNModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = (
            config.pad_token_id
            if hasattr(config, "pad_token_id") and config.pad_token_id is not None
            else 0
        )
        self.encoder = CYKNEncoder(config, args)
        self.pooler = BertPoolerCYK(config)

        logger.info("Init CYKN pretrain model")

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = extended_attention_mask.to(
            dtype=self.embeddings.word_embeddings.weight.dtype  # should be of same dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoder_output = self.encoder(
            embedding_output
        )
        sequence_output = encoder_output

        pooled_output = self.pooler(sequence_output)

        output = (
            sequence_output,
            pooled_output,
        )

        return output

class CYKNLMHeadModel(BertPreTrainedModel):
    
    def __init__(self, config, args):
        super(CYKNLMHeadModel, self).__init__(config)
        self.cyk = CYKNModel(config, args)

        self.cls = BertOnlyMLMHead(config, self.cyk.embeddings.word_embeddings.weight)
        self.init_weights()

    def forward(self, batch, output_attentions=False):
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[4]
        checkpoint_activations = False

        bert_output = self.cyk(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )
        sequence_output = bert_output[0]
        sequence_output = sequence_output.unsqueeze(-2).expand(-1,input_ids.shape[1] , -1)
        
        if masked_lm_labels is None:
            prediction_scores = self.cls(sequence_output)
            return prediction_scores

        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(
            -1
        )
        prediction_scores = self.cls(sequence_output, masked_token_indexes)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)

            outputs = (masked_lm_loss,)
            return outputs
        else:
            return prediction_scores
        
        
class CYKNForSequenceClassification(BertPreTrainedModel):
    
    def __init__(self, config, args=None):
        super(CYKNForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.cyk = CYKNModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerCYK(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):
        outputs = self.cyk(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        #pooled_output = self.pooler(outputs[0])
        
        pooled_output = self.dropout(outputs[0])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    

class CYKNForSequenceClassificationL(BertPreTrainedModel):
    

    def __init__(self, config, args=None):
        super(CYKNForSequenceClassificationL, self).__init__(config)
        self.num_labels = config.num_labels
        self.cyk = CYKNModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerCYK(config)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):

        outputs = self.cyk(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        pooled_output = self.pooler(outputs[0])
        #print(pooled_output.shape)
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
class CYKNForSequenceClassificationCondF(BertPreTrainedModel):
    def __init__(self, config, args):
        super(CYKNForSequenceClassificationCondF, self).__init__(config)
        self.cyk = CYKNModel(config, args)
        self.cls = BertOnlyMLMHead(config, self.cyk.embeddings.word_embeddings.weight)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerC(config)
        self.classifier = nn.Linear(config.vocab_size * 18, self.num_labels)

        self.init_weights()

    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,):
        sequence_output_list = []
        for i in range(20):
            masked_input_ids = copy.deepcopy(input_ids)
            masked_input_ids[:,i] = 103
            outputs = self.cyk(
            masked_input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
            )
            sequence_output_list.append(outputs[0].unsqueeze(-2))
        sequence_output = torch.cat(sequence_output_list, dim = 1)

        prediction_scores = F.softmax(self.cls(sequence_output), dim = -1)

        pooled_output = self.pooler(prediction_scores)
        
        
        
  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class CYKNForSequenceClassificationCondFN(BertPreTrainedModel):
    def __init__(self, config, args):
        super(CYKNForSequenceClassificationCondFN, self).__init__(config)
        self.cyk = CYKNModel(config, args)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerCYK(config)
        self.classifier = nn.Linear(config.hidden_size * 18, self.num_labels)

        self.init_weights()

    def forward(self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,):
        sequence_output_list = []
        for i in range(1,19):
            masked_input_ids = copy.deepcopy(input_ids)
            masked_input_ids[:,i] = 103
            outputs = self.cyk(
            masked_input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
            )
            sequence_output_list.append(outputs[0].unsqueeze(-2))
        sequence_output = torch.cat(sequence_output_list, dim = 1)
        
        sequence_output = torch.flatten(self.pooler(sequence_output), start_dim = 1)

        
        
        
  
        pooled_output = self.dropout(sequence_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class Max_tensorM(nn.Module):
    def __init__(self, in_features: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Max_tensorM, self).__init__()
        self.in_features = in_features
        self.pro_tensor = nn.Parameter(torch.Tensor(self.in_features, self.in_features, self.in_features))
        bound = math.sqrt(3.0) * 0.02
        nn.init.uniform_(self.pro_tensor, -bound, bound)
        


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_c, x_p = torch.chunk(input, 2, dim=-1)
        a_c = torch.einsum("btkn,pmn->btkpm", x_p, self.pro_tensor)
        a_cp = torch.einsum("btkpm,btkm->btkp", a_c, x_c)
        
        return a_cp

class CYKMEncoder(nn.Module):
    def __init__(self, config, args) -> None:
        super(CYKMEncoder, self).__init__()
        
        self.d_feat = config.hidden_size
        self.max_prob = Max_tensorM(config.hidden_size)
        self.max_l = Max_L()


    def shifta(self, x, dist, lr):
        B, l, n = x.shape
        new_vec = x.new_zeros((B, dist, n))#.to_device(x.device)
        #print(x.shape, new_vec.shape)
        if lr == 'l':
            return torch.cat([x, new_vec], dim = 1)
        else:
            return torch.cat([new_vec, x], dim = 1)


    
    def forward(self, x):
        B = x.shape[0]
        T = x.shape[1]
        v_list = []
        v_s_list = []
        v_list.append(x)
        v_s_list.append([self.shifta(x, 1, 'l'),self.shifta(x, 1, 'r')])
        for t in range(1,T):
            tp_list = []
            r_list = []
            for i in range(t):
                tp_list.append(torch.cat([v_s_list[i][1], v_s_list[t - 1 - i][0]], dim = -1)[:,i+1:T+i-t+1,:].unsqueeze(-2))
            tp_prob = torch.cat(tp_list, dim = -2)
            #print(tp_prob.shape)
            u = self.max_prob(tp_prob)

            curr_v = self.max_l(u)
            v_list.append(curr_v)
            v_s_list.append([self.shifta(curr_v,t+1, 'l'),self.shifta(curr_v, t+1, 'r')])
        

        v_feat = curr_v.squeeze(-2)
        return v_feat

class CYKMModel(BertPreTrainedModel):

    def __init__(self, config, args=None):
        super(CYKMModel, self).__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = (
            config.pad_token_id
            if hasattr(config, "pad_token_id") and config.pad_token_id is not None
            else 0
        )
        self.encoder = CYKMEncoder(config, args)
        self.pooler = BertPoolerCYK(config)

        logger.info("Init CYKM pretrain model")

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = extended_attention_mask.to(
            dtype=self.embeddings.weight.dtype  # should be of same dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids)[:,1:-1,:]

        encoder_output = self.encoder(
            embedding_output
        )
        sequence_output = encoder_output

        pooled_output = self.pooler(sequence_output)

        output = (
            sequence_output,
            pooled_output,
        )

        return output

class CYKMForSequenceClassification(BertPreTrainedModel):
    
    def __init__(self, config, args=None):
        super(CYKMForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.cyk = CYKMModel(config, args)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = BertPoolerCYK(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        checkpoint_activations=False,
        **kwargs,
    ):
        outputs = self.cyk(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=checkpoint_activations,
        )

        #pooled_output = self.pooler(outputs[0])
        
        pooled_output = self.dropout(outputs[0])
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )