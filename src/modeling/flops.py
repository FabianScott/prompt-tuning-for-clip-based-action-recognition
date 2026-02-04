# First part from https://github.com/openai/CLIP/issues/143, added the scaled dot product attention, native attention and linalg_vector_norm estimators

import torch
import typing
from collections import Counter

import torch.nn as nn
from fvcore.nn.jit_handles import batchnorm_flop_jit, matmul_flop_jit, generic_activation_jit, get_shape
prod = lambda shape: torch.tensor(shape).prod().item()

def generic_pooling_jit(name, multiplier=1):
    def pool_jit(inputs: typing.List[object], outputs: typing.List[object]) -> typing.Counter[str]:
        # Inputs[0] contains the shape of the input.
        input_shape = get_shape(inputs[0])
        output_shape = get_shape(outputs[0])
        assert 2 <= len(input_shape) <= 5, input_shape
        flop = prod(input_shape) + prod(output_shape)  # summing all elements + denominating in each for output
        flop_counter = Counter({name: flop * multiplier})
        return flop_counter

    return lambda inputs, outputs: pool_jit(inputs, outputs)

def softmax_jit(inputs: typing.List[object], outputs: typing.List[object]) -> typing.Counter[str]:
    input_shape = get_shape(inputs[0])
    output_shape = get_shape(outputs[0])
    flop = prod(input_shape) * 2 + prod(output_shape) # exponentiating & summing inputs + denominating in each batch
    flop_counter = Counter({"softmax": flop})
    return flop_counter

def bmm_flop_jit(inputs: typing.List[object], outputs: typing.List[object]) -> typing.Counter[str]:
    input1_shape = get_shape(inputs[0])
    input2_shape = get_shape(inputs[1])
    assert len(input1_shape) == len(input2_shape) == 3
    assert input1_shape[0] == input2_shape[0] and input1_shape[2] == input2_shape[1], [input1_shape, input2_shape]
    flop = prod(input1_shape) * input2_shape[-1]  # matmul of bnk * bkm -> bnm; flop = bnkm
    flop_counter = Counter({"bmm": flop})
    return flop_counter

def scaled_dot_product_attention_jit(inputs: typing.List[object], outputs: typing.List[object]) -> Counter:
    """
    Estimates FLOPs for aten::scaled_dot_product_attention.
    inputs[0]: Q tensor (B, n_heads, seq_len, d_head)
    inputs[1]: K tensor (B, n_heads, seq_len, d_head)
    inputs[2]: V tensor (B, n_heads, seq_len, d_head)
    """
    Q_shape = get_shape(inputs[0])
    K_shape = get_shape(inputs[1])
    V_shape = get_shape(inputs[2])
    B, h, L, d = Q_shape  # batch, heads, seq_len, head_dim

    # FLOPs for Q*K^T (B, h, L, L): 2 * L*L*d per head per batch
    flops_qk = 2 * B * h * L * L * d
    # FLOPs for scaling + softmax: assume 5*L*L per head per batch
    flops_softmax = B * h * L * L * 5
    # FLOPs for output = softmax*V (B, h, L, d): 2 * L*L*d
    flops_output = 2 * B * h * L * L * d

    total_flops = flops_qk + flops_softmax + flops_output
    return Counter({"scaled_dot_product_attention": total_flops})

def linalg_vector_norm_jit(inputs: typing.List[object], outputs: typing.List[object]) -> Counter:
    """
    Estimates FLOPs for aten::linalg_vector_norm.
    inputs[0]: tensor of shape (..., vector_dim)
    """
    input_shape = get_shape(inputs[0])
    num_elements = prod(input_shape)
    # 1 multiplication + 1 addition per element for sum-of-squares
    flop = 2 * num_elements
    return Counter({"linalg_vector_norm": flop})

def native_multi_head_attention_jit(inputs: list, outputs: list) -> Counter:
    """
    Estimates FLOPs for aten::_native_multi_head_attention.
    inputs[0]: query tensor (B, L, d_model)
    """
    B, L, d_model = get_shape(inputs[0])
    h = 12  # replace with actual number of heads if known
    d_head = d_model // h

    flops_linear = 4 * B * L * d_model**2      # Q,K,V + output projection
    flops_attn = 4 * B * L**2 * d_model + 5 * B * h * L**2

    total_flops = flops_linear + flops_attn
    return Counter({"_native_multi_head_attention": total_flops})

supported_ops={
    "aten::batch_norm": batchnorm_flop_jit,
    "aten::group_norm": batchnorm_flop_jit,
    "aten::layer_norm": batchnorm_flop_jit,
    "aten::add": generic_activation_jit("add"),
    "aten::sub": generic_activation_jit("sub"),
    "aten::mul": generic_activation_jit("mul"),
    "aten::div": generic_activation_jit("div"),
    "aten::sqrt": generic_activation_jit("sqrt"),
    "aten::sigmoid": generic_activation_jit("sigmoid"),
    "aten::sigmoid_": generic_activation_jit("sigmoid_"),
    "aten::relu": generic_activation_jit("relu"),
    "aten::relu_": generic_activation_jit("relu_"),
    "aten::gelu": generic_activation_jit("gelu"),
    "aten::add_": generic_activation_jit("add_"),
    "aten::sub_": generic_activation_jit("sub_"),
    "aten::mul_": generic_activation_jit("mul_"),
    "aten::div_": generic_activation_jit("div_"),
    "aten::sqrt_": generic_activation_jit("sqrt_"),
    "aten::adaptive_avg_pool2d": generic_pooling_jit("adaptive_avg_pool2d"),
    "aten::adaptive_max_pool2d": generic_pooling_jit("adaptive_max_pool2d"),
    "aten::avg_pool2d": generic_pooling_jit("avg_pool2d"),
    "aten::max_pool2d": generic_pooling_jit("max_pool2d"),
    "aten::bmm": bmm_flop_jit,
    "aten::mean": generic_pooling_jit("mean"),
    "aten::var": generic_pooling_jit("var", multiplier=3),  # subtracting mean, exponentiate, summing
    "aten::var_mean": generic_pooling_jit("mean_var", multiplier=4),
    "aten::softmax": softmax_jit,
    "aten::dropout": generic_activation_jit("dropout"),
    "aten::frobenius_norm": generic_pooling_jit("frobenius_norm"),
    "aten::scaled_dot_product_attention": scaled_dot_product_attention_jit,
    "aten::linalg_vector_norm": linalg_vector_norm_jit,
    "aten::_native_multi_head_attention": native_multi_head_attention_jit,
}


class CoOpForwardWrapper(nn.Module):
    def __init__(self, coop):
        super().__init__()
        self.coop = coop
        self.detach_params(self.coop.clip)
        self.detach_params(self.coop.context_learner)

    def detach_params(self, module):
        for param in module.parameters():
            param.requires_grad = False
            param.detach_()

    def forward(self, videos, video_masks):

        logits, _, _ = self.coop.forward(
            videos=videos,
            video_masks=video_masks,
            softmax=False,
        )
        return logits
