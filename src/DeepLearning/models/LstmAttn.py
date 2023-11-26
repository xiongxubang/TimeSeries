import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Rnn_EncDec import Seq2SeqEncoder, Seq2SeqAttentionDecoder
from layers.Embed import DataEmbedding



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.encoder = Seq2SeqEncoder(configs.d_model, num_hiddens=configs.d_model, num_layers=configs.e_layers, rnn_model=nn.LSTM)
        self.decoder = Seq2SeqAttentionDecoder(configs.d_model, num_hiddens=configs.d_model, num_layers=configs.e_layers, rnn_model=nn.LSTM, c_out=configs.c_out)
        

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.encoder(enc_out)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_state = self.decoder.init_state(enc_out)
        dec_out, _ = self.decoder(dec_out, dec_state)
        
        return dec_out[:, -self.pred_len:, :]