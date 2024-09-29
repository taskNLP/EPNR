import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from torch.nn import Parameter


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )
        self.mem_dim = channels
        self.head_dim = self.mem_dim//4
        self.convs = nn.ModuleList()
        for i in range(4):
            self.convs.append(nn.Conv2d(self.mem_dim + self.head_dim * i, self.head_dim, kernel_size=3, padding=1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = x
        cache_list = [outputs]
        for i in range(4):
            temp = self.convs[i](outputs)  # self loop
            cache_list.append(temp)
            outputs = torch.cat(cache_list, dim=1)
        outputs = outputs.permute(0, 2, 3, 1)
        return outputs


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.W_k = nn.Linear(input_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, input, entity_list):
        size = entity_list.size()
        x = input.unsqueeze(1).expand(-1, size[1], -1, -1)
        x = torch.masked_fill(x, entity_list.eq(0).unsqueeze(-1), 0)
        s = self.W_k(x).squeeze(-1)
        s = torch.masked_fill(s, entity_list.eq(0), -1e9)
        a = self.softmax(s)
        h = torch.bmm(a, input)
        return h


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers
        self.lstm_hid_size = config.lstm_hid_size

        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)

        self.dropout = nn.Dropout(config.emb_dropout)

        self.encoder = nn.LSTM(config.bert_hid_size, config.lstm_hid_size//2, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.attention = AttentionLayer(config.lstm_hid_size)

        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)
        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.conv = ConvolutionLayer(config.lstm_hid_size + config.dist_emb_size, config.conv_hid_size)

        self.linear = nn.Linear(in_features=config.conv_hid_size * 2, out_features=config.label_num)

    def forward(self, bert_inputs, pieces2word, pieces2entity, entity_mask, sent_length, grid_mask2d):

        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        word_reps = self.dropout(word_reps)

        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        outputs, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        outputs = self.attention(outputs, pieces2entity)

        cln = self.cln(outputs.unsqueeze(2), outputs)


        dis_embs = self.dis_embs(entity_mask)

        cln = torch.cat([cln, dis_embs], dim=-1)

        conv_inputs = torch.masked_fill(cln, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.conv(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)

        out = self.linear(conv_outputs)
        return out
