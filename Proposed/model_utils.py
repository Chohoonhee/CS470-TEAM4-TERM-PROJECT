""" Code for all the model submodules part
    of various model architecures. """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple
from torchvision.models import (mobilenet_v2, resnet18, resnet34, resnet50,
                                resnet101, resnet152)
RESNET_VERSION_TO_MODEL = {'resnet18': resnet18, 'resnet34': resnet34,
                           'resnet50': resnet50, 'resnet101': resnet101,
                           'resnet152': resnet152}


def trim_network_at_index(network: nn.Module, index: int = -1) -> nn.Module:
    """
    Returns a new network with all layers up to index from the back.
    :param network: Module to trim.
    :param index: Where to trim the network. Counted from the last layer.
    """
    assert index < 0, f"Param index must be negative. Received {index}."
    return nn.Sequential(*list(network.children())[:index])


def calculate_backbone_feature_dim(backbone, input_shape: Tuple[int, int, int]) -> int:
    """ Helper to calculate the shape of the fully-connected regression layer. """
    tensor = torch.ones(1, *input_shape)
    output_feat = backbone.forward(tensor)
    return output_feat.shape[-1]
class ResNetBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: resnet18, resnet34, resnet50, resnet101, resnet152.
    """

    def __init__(self, version: str):
        """
        Inits ResNetBackbone
        :param version: resnet version to use.
        """
        super().__init__()

        if version not in RESNET_VERSION_TO_MODEL:
            raise ValueError(f'Parameter version must be one of {list(RESNET_VERSION_TO_MODEL.keys())}'
                             f'. Received {version}.')

        self.backbone = trim_network_at_index(RESNET_VERSION_TO_MODEL[version](pretrained=True), -1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For resnet50,
            the shape is [batch_size, 2048].
        """
        backbone_features = self.backbone(input_tensor)
        return torch.flatten(backbone_features, start_dim=1)


class MobileNetBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: mobilenet_v2.
    """

    def __init__(self, version: str):
        """
        Inits MobileNetBackbone.
        :param version: mobilenet version to use.
        """
        super().__init__()

        if version != 'mobilenet_v2':
            raise NotImplementedError(f'Only mobilenet_v2 has been implemented. Received {version}.')

        self.backbone = trim_network_at_index(mobilenet_v2(pretrained=True), -1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width].
        :return: Tensor of shape [batch_size, n_convolution_filters]. For mobilenet_v2,
            the shape is [batch_size, 1280].
        """
        backbone_features = self.backbone(input_tensor)
        return backbone_features.mean([2, 3])




class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):

        d_k = k.size(-1)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d_k)
        # bs x n_head x seq_len x d_k @ # bs x n_head x d_k x seq_len
        # => bs x n_head x seq_len x seq_len

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9) # 1 x seq_len x seq_len

        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)
        # bs x n_head x seq_len x seq_len @ bs x n_head x seq_len x d_k
        # => bs x n_head x seq_len x d_k
        return output, attn


class Visual_Attention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, key_model, d_k, d_v, n_head=1, dropout=0.1):
        super(Visual_Attention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.softmax = nn.Softmax(dim=-1)
        self.Qw = nn.Linear(d_model, n_head * d_v)
        self.Kw = nn.Linear(key_model, n_head * d_v)
        self.Vw = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)  # sz_b = 1
        q = self.Qw(q)
        k = self.Kw(k)
        v = self.Vw(v)

        output = torch.matmul(q, k.transpose(0,1))
        output = self.dropout(self.softmax(output/d_k))
        output = torch.matmul(output, v)


        return output


class SelfAttention(nn.Module):
    ''' Multi-Head Attention module ''' 
    def __init__(self, d_model, d_k, d_v, n_head=1, dropout=0.1):
        super(SelfAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.Qw = nn.Linear(d_model, n_head * d_k)
        self.Kw = nn.Linear(d_model, n_head * d_k)
        self.Vw = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1) # sz_b = 1

        q = self.Qw(q).view(sz_b, len_q, n_head, d_k) # bs x seq_len x n_head x d_k
        k = self.Kw(k).view(sz_b, len_k, n_head, d_k)
        v = self.Vw(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # bs x n_head x seq_len x d_k
        
        if mask is not None: # bs x eq_len x seq_len
            mask = mask.unsqueeze(1)   # For head axis broadcasting => bs x 1 x seq_len x seq_len

        q, attn = self.attention(q, k, v, mask=mask) # bs x n_head x seq_len x d_k

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1) 
        # bs x seq_len x n_head x d_k 
        # => bs x seq_len x d_model
        output = self.dropout(self.fc(q))

        return output #, attn


class CrossModalAttention(nn.Module):
    """
    Crossmodal Attention Module from Show, Attend, and Tell
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim, att=True):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(CrossModalAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.att = att

    def forward(self, map_features, traj_encoding, episode_idx):
        """
        Forward propagation.
        :param map_features: encoded images, a tensor of dimension (agent_size, num_pixels, attention_dim)
        :param traj_encoding: previous decoder output, a tensor of dimension (agent_size, attention_dim)
        :return: attention weighted map encoding, weights
        """
        if self.att:
            att1 = self.encoder_att(map_features)  # (agent_size, num_pixels, attention_dim)
            att2 = self.decoder_att(traj_encoding)  # (agent_size, attention_dim)
            att = self.full_att(self.relu(att1[episode_idx].add_(att2.unsqueeze_(1))))  # (agent_size, num_pixels)
            
            alpha = self.softmax(att)  # (agent_size, num_pixels)
        else:
            alpha = torch.empty((episode_idx.size(0),map_features.size(1), 1), device=map_features.device).fill_(1/map_features.size(1))

        # att1: (agent_size, num_pixels, map_feat_dim) -> (agent_size, num_pixels, attention_dim) 
        # att2: (agent_size, num_pixels, traj_encoding_dim) -> (agent_size, attention_dim) 
        # att: (agent_size, num_pixels, attention_dim) + (agent_size, 1, attention_dim) -> (agent_size, num_pixels)
        # alpha: (agent_size, num_pixels)

        # alpha = torch.ones_like(alpha)
        attention_weighted_encoding = (map_features[episode_idx].mul_(alpha)).sum(dim=1)
        # (agent_size, num_pixels, encoder_dim) * (agent_size, num_pixels, 1) 
        # => (agent_size, num_pixels, encoder_dim)
        # => (agent_size, encoder_dim)

        return attention_weighted_encoding, alpha


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias, dilation=dilation),
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


        

class NewModelShallowCNN(nn.Module):

    def __init__(self, dropout=0.5, size=100):  # Output Size: 30 * 30
        super(NewModelShallowCNN, self).__init__()

        self.conv1 = conv2DBatchNormRelu(in_channels=3, n_filters=16, k_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=16, n_filters=16, k_size=3, stride=1, padding=1, dilation=1)
        self.pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv3 = conv2DBatchNormRelu(in_channels=16, n_filters=32, k_size=5, stride=1, padding=2, dilation=1)
        self.conv4 = conv2DBatchNormRelu(in_channels=32, n_filters=6, k_size=1, stride=1, padding=0, dilation=1)

        self.dropout = nn.Dropout(p=dropout)
        self.upsample = nn.Upsample(size=size, mode='bilinear')

    def forward(self, image, size=60):

        x = self.conv1(image) # 64 >> 64
        x = self.conv2(x) # 64 >> 64
        x = self.pool1(x) # 64 >> 32
        x = self.conv3(x) # 32 >> 32
        local_ = self.conv4(x) # 32 >> 32
        global_ = self.dropout(x)

        local_ = self.upsample(local_)

        return local_, global_

