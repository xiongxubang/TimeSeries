import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


#@save
class Seq2SeqEncoder(Encoder):
    def __init__(self, embed_size, num_hiddens, num_layers,
                 dropout=0, rnn_model=nn.GRU, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.rnn = rnn_model(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输入'X'的形状：(batch_size,num_steps,embed_size)，已经做了dataEmbedding
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

class Seq2SeqDecoder(Decoder):
    def __init__(self, embed_size, num_hiddens, num_layers, c_out,
                 dropout=0, rnn_model=nn.GRU, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.rnn = rnn_model(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout, bidirectional=False)
        self.dense = nn.Linear(num_hiddens, c_out)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输入'X'的形状：(batch_size,num_steps,embed_size)
        # 改变'X'的形状
        X = X.permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        if type(self.rnn) == nn.LSTM:
            context = state[-1][1].repeat(X.shape[0], 1, 1)
        else:
            context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state