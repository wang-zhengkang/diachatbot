# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import random

import numpy as np
import torch
import torch.nn as nn

if torch.cuda.is_available():
    DEVICES = [0, 1]
    # DEVICES = [0]


def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return eps.mul(std) + mu


def batch_gather_3_1(inputs, dim):
    """
    Args:
        inputs (batchsz, sen_len, embed_dim)
        dim (batchsz)
    Returns:
        output (batch, embed_dim)
    """
    a = torch.arange(dim.shape[0])
    b = dim.view(-1) - 1
    output = inputs[a, b, :]
    return output


def batch_gather_4_2(inputs, dim):
    """
    Args:
        inputs (batchsz, sen_len, word_len, embed_dim)
        dim (batchsz, sen_len)
    Returns:
        output (batch, sen_len, embed_dim)
    """
    a = torch.arange(dim.shape[0])
    a = a.unsqueeze(1).expand(-1, dim.shape[1]).contiguous().view(-1)
    b = torch.arange(dim.shape[1])
    b = b.unsqueeze(0).expand(dim.shape[0], -1).contiguous().view(-1)
    c = dim.view(-1) - 1
    output = inputs[a, b, c, :].view(dim.shape[0], dim.shape[1], -1)
    return output


class VHUS(nn.Module):
    def __init__(self, cfg, voc_goal_size, voc_usr_size, voc_sys_size):
        super(VHUS, self).__init__()

        self.goal_encoder = Encoder(voc_goal_size, cfg['eu_dim'], cfg['hu_dim'])
        self.sys_encoder = Encoder(voc_sys_size, cfg['eu_dim'], cfg['hu_dim'])
        self.context_encoder = nn.GRU(cfg['hu_dim'], cfg['hu_dim'], batch_first=True)

        self.mu_net = nn.Linear(cfg['hu_dim'], cfg['hu_dim'])
        self.logvar_net = nn.Linear(cfg['hu_dim'], cfg['hu_dim'])
        self.mu_net_last = nn.Linear(cfg['hu_dim'], cfg['hu_dim'])
        self.logvar_net_last = nn.Linear(cfg['hu_dim'], cfg['hu_dim'])
        self.concat_net = nn.Linear(cfg['hu_dim'] * 2, cfg['hu_dim'])

        self.terminated_net = nn.Sequential(nn.Linear(cfg['hu_dim'], cfg['hu_dim']),
                                            nn.ReLU(),
                                            nn.Linear(cfg['hu_dim'], 1))
        self.usr_decoder = Decoder(voc_usr_size, cfg['max_ulen'], cfg['eu_dim'], cfg['hu_dim'])

    def forward(self, goals, goals_length, posts, posts_length, origin_responses=None):
        goal_output, _ = self.goal_encoder(goals)  # [B, G, H]
        goal_h = batch_gather_3_1(goal_output, goals_length)  # [B, H]

        batchsz, max_sen, max_word = posts.shape
        post_flat = posts.view(batchsz * max_sen, max_word)
        post_output_flat, _ = self.sys_encoder(post_flat)
        post_output = post_output_flat.view(batchsz, max_sen, max_word, -1)  # [B, S, P, H]
        post_h = batch_gather_4_2(post_output, posts_length)  # [B, S, H]

        context_output, _ = self.context_encoder(post_h, goal_h.unsqueeze(0))  # [B, S, H]
        posts_sen_length = posts_length.gt(0).sum(1)  # [B]

        context = batch_gather_3_1(context_output, posts_sen_length)  # [B, H]
        mu, logvar = self.mu_net(context), self.logvar_net(context)
        last_context = batch_gather_3_1(context_output, posts_sen_length - 1)
        mu_last, logvar_last = self.mu_net_last(last_context), self.logvar_net_last(last_context)
        z = reparameterize(mu_last, logvar_last)
        hidden = self.concat_net(torch.cat([context, z], dim=1))

        teacher = 1 if origin_responses is not None else 0
        a_weights, _, _ = self.usr_decoder(inputs=origin_responses, encoder_hidden=hidden.unsqueeze(0), \
                                           teacher_forcing_ratio=teacher)
        t_weights = self.terminated_net(context).squeeze(1)

        return a_weights, t_weights, (mu_last, logvar_last, mu, logvar)

    def select_action(self, goal, goal_length, post, post_length):
        """
        :param goal: [goal_len]
        :param goal_length: []
        :param post: [sen_len, word_len]
        :param post_length: [sen_len]
        :return: [act_len], [1]
        """
        goal, goal_length, post, post_length = goal.unsqueeze(0), \
            goal_length.unsqueeze(0), post.unsqueeze(0), \
            post_length.unsqueeze(0)

        a_weights, t_weights, _ = self.forward(goal, goal_length, post, post_length)
        usr_a = []
        for a_weight in a_weights:
            a = a_weight.argmax(1).item()
            if a == self.usr_decoder.eos_id:
                break
            usr_a.append(a)
        terminated = t_weights.ge(0).item()

        return usr_a, terminated


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, input_dropout_p=0, dropout_p=0, n_layers=1,
                 rnn_cell='GRU', variable_lengths=False, embedding=None, update_embedding=True):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell == 'LSTM':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'GRU':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of
              the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the
              features in the hidden state h
        """
        self.rnn.flatten_parameters()
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class Decoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, embed_size, hidden_size, sos_id=2, eos_id=3, n_layers=1, rnn_cell='GRU',
                 input_dropout_p=0, dropout_p=0, use_attention=False):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell == 'LSTM':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'GRU':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
        self.rnn = self.rnn_cell(embed_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, embed_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size,
                                                                                                           output_size,
                                                                                                           -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=torch.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        self.rnn.flatten_parameters()
        if self.use_attention:
            ret_dict[Decoder.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn, infer=False):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            if infer and not step:
                symbols = torch.cat((decoder_outputs[-1][:, :self.eos_id],
                                     decoder_outputs[-1][:, (self.eos_id + 1):]), 1).topk(1)[1]
                symbols.add_(symbols.ge(self.eos_id).long())
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                              encoder_outputs,
                                                                              function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn, infer=True)
                decoder_input = symbols

        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict  # NLLLoss

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda(device=DEVICES[0])
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length

# class LayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         """
#         Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(LayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias
        
class Attention(nn.Module):
    pass
#     def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
#         super(Attention, self).__init__()
#         if hidden_size % num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (hidden_size, num_attention_heads))
#         self.num_attention_heads = num_attention_heads
#         self.attention_head_size = int(hidden_size / num_attention_heads)
#         self.all_head_size = hidden_size

#         self.query = nn.Linear(input_size, self.all_head_size)
#         self.key = nn.Linear(input_size, self.all_head_size)
#         self.value = nn.Linear(input_size, self.all_head_size)

#         self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

#         # 做完self-attention 做一个前馈全连接 LayerNorm 输出
#         self.dense = nn.Linear(hidden_size, hidden_size)
#         self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
#         self.out_dropout = nn.Dropout(hidden_dropout_prob)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, input_tensor):
#         mixed_query_layer = self.query(input_tensor)
#         mixed_key_layer = self.key(input_tensor)
#         mixed_value_layer = self.value(input_tensor)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         # [batch_size heads seq_len seq_len] scores
#         # [batch_size 1 1 seq_len]

#         # attention_scores = attention_scores + attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         # Fixme
#         attention_probs = self.attn_dropout(attention_probs)
#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         hidden_states = self.dense(context_layer)
#         hidden_states = self.out_dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)

#         return hidden_states
