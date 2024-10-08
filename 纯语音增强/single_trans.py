import torch
import os

import torch.nn as nn
import torch
import copy
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
# from SubLayers import MultiHeadAttention

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead=4, bidirectional=True, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_c = MultiheadAttention(128, nhead, dropout=dropout)
        # 可改多头 对头时间的维度
        # self.self_attn = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=, d_v=, dropout=dropout)#n_head, d_model, d_k, d_v
        # conv-ffn
        self.pad = nn.ConstantPad1d((1, 1), value=0.)
        self.Conv1D_3 = nn.Conv1d(32, 64, kernel_size=3)
        self.BN = nn.BatchNorm1d(64)  # (C)
        self.PReLU = nn.PReLU()
        self.Conv1D_1 = nn.Conv1d(64, 32, kernel_size=1)

        # FFN
        ## self.linear1 = Linear(d_model, dim_feedforward)
        ## self.linear2 = Linear(dim_feedforward, d_model)

        # self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        # self.dropout = Dropout(dropout)
        # if bidirectional:
        #     self.linear2 = Linear(d_model * 2 * 2, d_model)
        # else:
        #     self.linear2 = Linear(d_model * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self.arfa = nn.Parameter(torch.ones(1))  # 一次一个
        self.beta = nn.Parameter(torch.ones(1))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        ### type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # 可改多头
        src_1 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_2 = self.self_attn_c(src.transpose(1, 2), src.transpose(1, 2), src.transpose(1, 2), attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)[0]
        src_2 = src_2.transpose(1, 2)

        src2 = self.arfa * src_1 + self.beta * src_2

        src = src + self.dropout1(src2)
        src = self.norm1(src)  # [32 ,256, 32]

        # Conv-FFN 输出为
        # src2 = src.view(32, 1, 256, -1).contiguous().permute(1, 3, 2, 0)
        src2 = self.Conv1D_1(self.PReLU(self.BN(self.Conv1D_3(self.pad(src.transpose(1, 2))))))
        src2 = src2.transpose(1, 2)

        # #src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # #FFN
        # self.gru.flatten_parameters()
        # out, h_n = self.gru(src)
        # del h_n
        # src2 = self.linear2(self.dropout(self.activation(out)))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
