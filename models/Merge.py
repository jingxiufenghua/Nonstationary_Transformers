import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from models import Autoformer,Informer,Transformer
from ns_models import ns_Autoformer,ns_Informer,ns_Transformer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.autoformer = Autoformer.Model(configs).float()
        self.autoformer2 = Autoformer.Model(configs).float()
        self.informer = Informer.Model(configs).float()
        self.informer2 = Informer.Model(configs).float()
        self.ns_autoformer = ns_Autoformer.Model(configs).float()
        self.ns_autoformer2 = ns_Autoformer.Model(configs).float()
        self.ns_Informer = ns_Informer.Model(configs).float()
        self.ns_Informer2 = ns_Informer.Model(configs).float()

        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        # self.lstm = nn.LSTM()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        first_tree = self.ns_Informer(x_enc, x_mark_enc,x_dec,x_mark_dec)
        second_tree = self.ns_Informer2(x_enc, x_mark_enc,x_dec,x_mark_dec)
        end_out = self.projection(torch.div(first_tree + second_tree,2))
        return end_out[:, -self.pred_len:, :]  # [B, L, D]

