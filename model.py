""" Transformer architecture for English -> French translation. """


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== ALL THE CLASSES ARE BELOW THIS LINE =====

class Embedding(nn.Module):

    def __init__(self, vocab_size, d_model):
      super().__init__()
      self.d_model = d_model
      self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
      return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
       super().__init__()

       pe = torch.zeros(max_len, d_model) # creates an empty matrix to store the positional values.

       position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Creates the position number of every token.

       div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

       pe[:, 0::2] = torch.sin(position * div_term)
       pe[:, 1::2] = torch.cos(position * div_term)
       pe = pe.unsqueeze(0)

       self.register_buffer('pe', pe)

    def forward(self, x):

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class ScaledDotProductAttention(nn.Module):

  def __init__(self, d_k):
    super().__init__()
    self.d_k = d_k

  def forward(self, q, k, v, mask=None):
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

    if mask is not None:
      attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(attention_scores, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):

  def __init__(self, d_model, num_heads):
    super().__init__()
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)

    self.W_o = nn.Linear(d_model, d_model)

    self.attention = ScaledDotProductAttention(self.d_k)

  def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)

    q = self.W_q(query)
    k = self.W_k(key)
    v = self.W_v(value)

    # splits each Q, K, and V vector into multiple attention heads, where each head has dimension d_k

    q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    output, attention_weights = self.attention(q, k, v, mask)

    # combines the outputs of all attention heads back into a single tensor of dimension d_model.

    output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    output = self.W_o(output)

    return output, attention_weights


class LayerNormalization(nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(d_model))
    self.beta = nn.Parameter(torch.zeros(d_model))

    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False) # Using population variance

    output = (x - mean) / torch.sqrt(variance + self.eps)

    output = self.gamma * output + self.beta

    return output

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)


  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = self.dropout(x)
    x = self.linear2(x)
    return x

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super().__init__()

    self.multihead_attention = MultiHeadAttention(d_model, num_heads)

    self.norm1 = LayerNormalization(d_model)

    self.norm2 = LayerNormalization(d_model)

    self.feed_forward = FeedForward(d_model, d_ff)

    self.dropout1 = nn.Dropout(dropout)

    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x, mask=None):

    attention_output, _ = self.multihead_attention(x, x, x, mask)

    attention_output = self.dropout1(attention_output)

    x = self.norm1(x + attention_output)

    feed_forward_output = self.feed_forward(x)

    feed_forward_output = self.dropout2(feed_forward_output)

    x = self.norm2(x + feed_forward_output)

    return x

class Encoder(nn.Module):
  def __init__(self, num_layers, vocab_size, max_len, d_model, num_heads, d_ff, dropout=0.1):
    super().__init__()

    self.embedding = Embedding(vocab_size, d_model)

    self.positional_encoding = PositionalEncoding(d_model, max_len)

    self.dropout = nn.Dropout(dropout)

    self.layers = nn.ModuleList([
        EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

  def forward(self, x, mask=None):
    x = self.embedding(x)

    x = self.positional_encoding(x)

    x = self.dropout(x)

    for layer in self.layers:
      x = layer(x, mask)

    return x

def create_padding_mask(x, pad_idx):
    return (x != pad_idx).unsqueeze(1).unsqueeze(2)


def generate_look_ahead_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super().__init__()

    self.masked_multihead_attention = MultiHeadAttention(d_model, num_heads)

    self.cross_attention = MultiHeadAttention(d_model, num_heads)

    self.feed_forward = FeedForward(d_model, d_ff)

    self.norm1 = LayerNormalization(d_model)

    self.norm2 = LayerNormalization(d_model)

    self.norm3 = LayerNormalization(d_model)

    self.dropout1 = nn.Dropout(dropout)

    self.dropout2 = nn.Dropout(dropout)

    self.dropout3 = nn.Dropout(dropout)


  def forward(self, x, encoder_output, look_ahead_mask=None, src_padding_mask=None):

    masked_attention_output, _ = self.masked_multihead_attention(x, x, x, look_ahead_mask)

    masked_attention_output = self.dropout1(masked_attention_output)

    x = self.norm1(x + masked_attention_output)

    cross_attention_output, attn_weights = self.cross_attention(x, encoder_output, encoder_output, src_padding_mask)

    self.last_attn_weights = attn_weights

    cross_attention_output = self.dropout2(cross_attention_output)

    x = self.norm2(x + cross_attention_output)

    feed_forward_output = self.feed_forward(x)

    feed_forward_output = self.dropout3(feed_forward_output)

    x = self.norm3(x + feed_forward_output)

    return x


class Decoder(nn.Module):
  def __init__(self, num_layers, vocab_size, max_len, d_model, num_heads, d_ff, dropout=0.1):
    super().__init__()

    self.embedding = Embedding(vocab_size, d_model)

    self.positional_encoding = PositionalEncoding(d_model, max_len)

    self.dropout = nn.Dropout(dropout)

    self.layers = nn.ModuleList([
        DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

  def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
    x = self.embedding(x)

    x = self.positional_encoding(x)

    x = self.dropout(x)

    for layer in self.layers:
      x = layer(x, encoder_output, look_ahead_mask, padding_mask)

    return x

class Transformer(nn.Module):
  def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len, trg_max_len, d_model, num_heads, d_ff):
    super().__init__()

    self.encoder = Encoder(num_layers, src_vocab_size, src_max_len, d_model, num_heads, d_ff)

    self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len, d_model, num_heads, d_ff)

    self.fc_out = nn.Linear(d_model, trg_vocab_size)

  def forward(self, src, trg):

    src_padding_mask = create_padding_mask(src, english_vocab["<pad>"])

    trg_padding_mask = create_padding_mask(trg, french_vocab["<pad>"])

    look_ahead_mask = generate_look_ahead_mask(trg.size(1), trg.device)

    combined_mask = trg_padding_mask & look_ahead_mask

    encoder_output = self.encoder(src, src_padding_mask)

    decoder_output = self.decoder(trg, encoder_output, combined_mask, src_padding_mask)

    output = self.fc_out(decoder_output)

    return output


# ===== ALL THE CLASSES ARE ABOVE THIS LINE =====


def build_model(src_vocab_size, trg_vocab_size, src_max_len, trg_max_len, device):
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_max_len=src_max_len,
        trg_max_len=trg_max_len,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024
    ).to(device)
    return model
