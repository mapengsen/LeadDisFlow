"""
Implementation of the decorator using a Encoder-Decoder architecture.
"""
import math

import torch
import torch.nn as tnn
import torch.nn.utils.rnn as tnnur


# 加入噪声正则化
import torch.nn as nn

class Encoder(nn.Module):
    """
    Simple bidirectional RNN encoder implementation.
    """

    def __init__(self, num_layers, num_dimensions, vocabulary_size, dropout, reduction_ratio=16):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size
        self.dropout = dropout

        self._embedding = nn.Sequential(
            nn.Embedding(self.vocabulary_size, self.num_dimensions),
            nn.Dropout(dropout)
        )

        # 使用LSTM和GRU
        self._rnn_lstm = nn.LSTM(self.num_dimensions, self.num_dimensions, self.num_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

        self._rnn_gru = nn.GRU(self.num_dimensions, self.num_dimensions, self.num_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

        # Squeeze-and-Excitation block
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.num_dimensions*2 , self.num_dimensions  // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.num_dimensions  // reduction_ratio, self.num_dimensions *2, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, padded_seqs, seq_lengths, std=0.1):
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the sequences (batch, seq).
        :param seq_lengths: The lengths of the sequences (for packed sequences).
        :param std: The standard deviation of the Gaussian noise.
        :return : A tensor with all the output values for each step and the two hidden states.
        """
        batch_size = padded_seqs.size(0)
        max_seq_size = padded_seqs.size(1)
        hidden_state_lstm = self._initialize_hidden_state(batch_size)
        hidden_state_gru = self._initialize_hidden_state(batch_size)

        padded_seqs = self._embedding(padded_seqs)

        # LSTM计算
        hs_h_lstm, hs_c_lstm = (hidden_state_lstm, hidden_state_lstm.clone().detach())
        packed_seqs = nn.utils.rnn.pack_padded_sequence(padded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_seqs, (hs_h_lstm, hs_c_lstm) = self._rnn_lstm(packed_seqs, (hs_h_lstm, hs_c_lstm))
        padded_seqs_lstm, _ = nn.utils.rnn.pad_packed_sequence(packed_seqs, batch_first=True)

        # GRU计算
        hs_h_gru = hidden_state_gru
        packed_seqs = nn.utils.rnn.pack_padded_sequence(padded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_seqs, hs_h_gru = self._rnn_gru(packed_seqs, hs_h_gru)
        padded_seqs_gru, _ = nn.utils.rnn.pad_packed_sequence(packed_seqs, batch_first=True)

        # 合并LSTM和GRU的输出
        padded_seqs = torch.add(padded_seqs_lstm, padded_seqs_gru)

        # apply Squeeze-and-Excitation block
        se_tensor = padded_seqs.permute(0, 2, 1)  # (batch, dim, seq)
        se_tensor = self.se_block(se_tensor)      # (batch, dim, 1)
        padded_seqs = padded_seqs * se_tensor.permute(0, 2, 1)  # (batch, seq, dim*2)

        # add Gaussian noise to hidden state
        hs_h_lstm_noise = hs_h_lstm + torch.randn_like(hs_h_lstm) * std
        hs_c_lstm_noise = hs_c_lstm + torch.randn_like(hs_c_lstm) * std

        hs_h_gru_noise = hs_h_gru + torch.randn_like(hs_h_gru) * std

        # sum up bidirectional layers and collapse
        hs_h_lstm_noise = hs_h_lstm_noise.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1).squeeze()
        hs_c_lstm_noise = hs_c_lstm_noise.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1).squeeze()

        hs_h_gru_noise = hs_h_gru_noise.view(self.num_layers, 2, batch_size, self.num_dimensions).sum(dim=1).squeeze()

        padded_seqs = padded_seqs.view(batch_size, max_seq_size, 2, self.num_dimensions).sum(dim=2).squeeze()

        return padded_seqs, (torch.add(hs_h_lstm_noise, hs_h_gru_noise), hs_c_lstm_noise)

    def _initialize_hidden_state(self, batch_size):
        # 初始化一个全零张量作为初始隐藏状态
        return torch.zeros(self.num_layers * 2, batch_size, self.num_dimensions).cuda()

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return {
            "num_layers": self.num_layers,
            "num_dimensions": self.num_dimensions,
            "vocabulary_size": self.vocabulary_size,
            "dropout": self.dropout
        }


class AttentionLayer(tnn.Module):

    def __init__(self, num_dimensions):
        super(AttentionLayer, self).__init__()

        self.num_dimensions = num_dimensions

        self._attention_linear = tnn.Sequential(
            tnn.Linear(self.num_dimensions*2, self.num_dimensions),
            tnn.Tanh()
        )

    def forward(self, padded_seqs, encoder_padded_seqs, decoder_mask):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_padded_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param decoder_mask: A tensor that represents the encoded input mask.
        :return : Two tensors: one with the modified logits and another with the attention weights.
        """
        # scaled dot-product
        # (batch, seq_d, 1, dim)*(batch, 1, seq_e, dim) => (batch, seq_d, seq_e*)
        attention_weights = (padded_seqs.unsqueeze(dim=2)*encoder_padded_seqs.unsqueeze(dim=1))\
            .sum(dim=3).div(math.sqrt(self.num_dimensions))\
            .softmax(dim=2)
        # (batch, seq_d, seq_e*)@(batch, seq_e, dim) => (batch, seq_d, dim)
        attention_context = attention_weights.bmm(encoder_padded_seqs)
        attention_masked = self._attention_linear(torch.cat([padded_seqs, attention_context], dim=2))*decoder_mask
        return (attention_masked, attention_weights)


class Decoder(tnn.Module):

    def __init__(self, num_layers, num_dimensions, vocabulary_size, dropout):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.num_dimensions = num_dimensions
        self.vocabulary_size = vocabulary_size
        self.dropout = dropout

        self._embedding = tnn.Sequential(
            tnn.Embedding(self.vocabulary_size, self.num_dimensions),
            tnn.Dropout(dropout)
        )
        self._rnn = tnn.LSTM(self.num_dimensions, self.num_dimensions, self.num_layers, batch_first=True, dropout=self.dropout, bidirectional=False)

        self._attention = AttentionLayer(self.num_dimensions)

        self._linear = tnn.Linear(self.num_dimensions, self.vocabulary_size)  # just to redimension

    def forward(self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param padded_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param seq_lengths: A list with the length of each output sequence.
        :param encoder_padded_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param hidden_states: The hidden states from the encoder.
        :return : Three tensors: The output logits, the hidden states of the decoder and the attention weights.
        """
        padded_encoded_seqs = self._embedding(padded_seqs)
        packed_encoded_seqs = tnnur.pack_padded_sequence(padded_encoded_seqs, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_encoded_seqs, hidden_states = self._rnn(packed_encoded_seqs, hidden_states)
        padded_encoded_seqs, _ = tnnur.pad_packed_sequence(packed_encoded_seqs, batch_first=True)  # (batch, seq, dim)

        mask = (padded_encoded_seqs[:, :, 0] != 0).unsqueeze(dim=-1).type(torch.float)
        attn_padded_encoded_seqs, attention_weights = self._attention(padded_encoded_seqs, encoder_padded_seqs, mask)
        logits = self._linear(attn_padded_encoded_seqs)*mask  # (batch, seq, voc_size)
        return logits, hidden_states, attention_weights

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return {
            "num_layers": self.num_layers,
            "num_dimensions": self.num_dimensions,
            "vocabulary_size": self.vocabulary_size,
            "dropout": self.dropout
        }


class Decorator(tnn.Module):
    """
    An encoder-decoder that decorates scaffolds.
    """

    def __init__(self, encoder_params, decoder_params):
        super(Decorator, self).__init__()

        self._encoder = Encoder(**encoder_params)
        self._decoder = Decoder(**decoder_params)

    def forward(self, encoder_seqs, encoder_seq_lengths, decoder_seqs, decoder_seq_lengths):  # pylint: disable=arguments-differ
        """
        Performs the forward pass.
        :param encoder_seqs: A tensor with the output sequences (batch, seq_d, dim).
        :param encoder_seq_lengths: A list with the length of each input sequence.
        :param decoder_seqs: A tensor with the encoded input scaffold sequences (batch, seq_e, dim).
        :param decoder_seq_lengths: The lengths of the decoder sequences.
        :return : The output logits as a tensor (batch, seq_d, dim).
        """
        encoder_padded_seqs, hidden_states = self.forward_encoder(encoder_seqs, encoder_seq_lengths)
        logits, _, attention_weights = self.forward_decoder(decoder_seqs, decoder_seq_lengths, encoder_padded_seqs, hidden_states)
        return logits, attention_weights

    def forward_encoder(self, padded_seqs, seq_lengths):
        """
        Does a forward pass only of the encoder.
        :param padded_seqs: The data to feed the encoder.
        :param seq_lengths: The length of each sequence in the batch.
        :return : Returns a tuple with (encoded_seqs, hidden_states)
        """
        return self._encoder(padded_seqs, seq_lengths)

    def forward_decoder(self, padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states):
        """
        Does a forward pass only of the decoder.
        :param hidden_states: The hidden states from the encoder.
        :param padded_seqs: The data to feed to the decoder.
        :param seq_lengths: The length of each sequence in the batch.
        :return : Returns the logits, the hidden state for each element of the sequence passed and the attention weights.
        """
        return self._decoder(padded_seqs, seq_lengths, encoder_padded_seqs, hidden_states)

    def get_params(self):
        """
        Obtains the params for the network.
        :return : A dict with the params.
        """
        return {
            "encoder_params": self._encoder.get_params(),
            "decoder_params": self._decoder.get_params()
        }
