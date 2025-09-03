# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""
ms_deform_attn_func
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from matplotlib.pylab import size
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from rfdetr import config


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """"for debug and test only, need to use cuda version instead
    """
    # B, n_heads, head_dim, N
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads, L, P, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H, W) in enumerate(value_spatial_shapes):
        # B, n_heads, head_dim, H, W
        value_l_ = value_list[lid_].view(B * n_heads, head_dim, H, W)
        # B, Len_q, n_heads, P, 2 -> B, n_heads, Len_q, P, 2 -> B*n_heads, Len_q, P, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # B*n_heads, head_dim, Len_q, P
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (B, Len_q, n_heads, L * P) -> (B, n_heads, Len_q, L, P) -> (B*n_heads, 1, Len_q, L*P)
    attention_weights = attention_weights.transpose(1, 2).reshape(B * n_heads, 1, Len_q, L * P)
    # B*n_heads, head_dim, Len_q, L*P
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()

def ms_deform_attn_core_pytorch_tflite(value, value_spatial_shapes, sampling_locations, attention_weights):
    """"for debug and test only, need to use cuda version instead
    """
    #print('called ms_deform_attn_core_pytorch_tflite')
    # B, n_heads, head_dim, N
    n_heads, head_dim, _ = value.shape
    Len_q, n_heads, L, P, _ = sampling_locations.shape
    ###TODO rewrite using slicing
    #use global model_spatial_shapes to avoid tensor ambiguity from torch Dynamo
    model_spatial_shapes = config.model_spatial_shapes
    # Use pure Python calculations instead of tensors
    split_sizes = [(size[0] * size[1]) for size in model_spatial_shapes]
    cumulative_indices = [0]
    for size in split_sizes:
        cumulative_indices.append(cumulative_indices[-1] + size)
    
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_ in range(len(model_spatial_shapes)):
        H, W = model_spatial_shapes[lid_]
        start_idx = cumulative_indices[lid_]
        end_idx = cumulative_indices[lid_ + 1]

        # B, n_heads, head_dim, H, W
        value_l_ = value[:, :, start_idx:end_idx] 
        value_l_ = value_l_.view(n_heads, head_dim, H, W)
        # B, Len_q, n_heads, P, 2 -> B, n_heads, Len_q, P, 2 -> B*n_heads, Len_q, P, 2
        sampling_grid_l_ = sampling_grids[:, :, lid_].transpose(0, 1)
        # B*n_heads, head_dim, Len_q, P
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (B, Len_q, n_heads, L * P) -> (B, n_heads, Len_q, L, P) -> (B*n_heads, 1, Len_q, L*P)
    attention_weights = attention_weights.transpose(0, 1).reshape(n_heads, 1, Len_q, L * P)
    # B*n_heads, head_dim, Len_q, L*P
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(n_heads * head_dim, Len_q)
    return output.transpose(0, 1).contiguous()