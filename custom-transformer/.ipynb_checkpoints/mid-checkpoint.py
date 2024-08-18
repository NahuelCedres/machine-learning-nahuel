### Libraries ###
import torch.nn as nn
from low import PositionalEncoding, Head, FeedForward, MultiHeadAttention


class Encoder(nn.Module):
    """
    Encoder.
    The Encoder is a key part of the Transformer model, consisting of multiple layers of self-attention and feedforward networks.
    
    Params:
    - dmodel: Number of dimensions for each token.
    - num_head: Number of attention heads.
    - dropout: Dropout rate for the dropout layers.
    """
    def __init__(self, dmodel, num_head, dropout):
        super(Encoder, self).__init__()

        #MultiHeadAttention
        self.mha = MultiHeadAttention(dmodel, num_head, dropout)
        #FeedForward
        self.ff = FeedForward(dmodel, dropout)
        #Normalization Layers
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        #dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
         """
        Forward pass of the Encoder.
        
        The input sequence goes through a self-attention layer followed by a feedforward network, both of which are followed by layer normalization and dropout.

        Params:
        - x: Input tensor of shape (batch_size, seq_length, dmodel).
        - mask: Masking tensor to prevent the model from attending to certain positions (e.g., padding).

        Returns:
        - x: Output tensor of the Encoder with the same shape as the input.
        """
        
        #First sub-layer
        x_out = self.mha( x,  x, x, mask)
        x = self.norm1(x + self.dropout(x_out))

        #Second sub-layer
        x_out = self.ff(x)
        x = self.norm1(x + self.dropout(x_out))
        
        return x


class Decoder(nn.Module):
    """
    Decoder.
    The Decoder is a key part of the Transformer model, designed to generate output sequences based on the encoded input sequence. 
    It consists of layers of masked self-attention, cross-attention with the encoder, and feedforward networks.
    
    Params:
    - dmodel: Number of dimensions for each token.
    - num_head: Number of attention heads.
    - dropout: Dropout rate for the dropout layers.
    """
    def __init__(self, dmodel, num_head, dropout):
        super(Decoder, self).__init__()

        #MultiHeadAttentions
        self.mha_mask = MultiHeadAttention(dmodel, num_head, dropout)
        self.mha_ = MultiHeadAttention(dmodel, num_head, dropout)
        #FeedForward
        self.ff_d = FeedForward(dmodel, dropout)
        #Normalization Layers
        self.norm1_d = nn.LayerNorm(dmodel)
        self.norm2_d = nn.LayerNorm(dmodel)
        self.norm3_d = nn.LayerNorm(dmodel)
        #Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):

        """
        Forward pass of the Decoder.

        The input sequence goes through a series of attention layers (masked self-attention and cross-attention with the encoder output),
        followed by a feedforward network. Each step is followed by layer normalization and dropout.

        Params:
        - x: Input tensor of shape (batch_size, seq_length, dmodel) representing the target sequence.
        - enc_out: Output tensor from the encoder of shape (batch_size, seq_length, dmodel).
        - src_mask: Masking tensor for the source sequence to prevent attention to padding tokens.
        - tgt_mask: Masking tensor for the target sequence to prevent the decoder from attending to future tokens.

        Returns:
        - x: Output tensor of the Decoder with the same shape as the input.
        """

        #First sub-layer
        x_out = self.mha_mask(x, x, x, tgt_mask)
        x = self.norm1_d(x + self.dropout(x_out))

        #Second sub-layer
        x_out = self.mha_(x, enc_out, enc_out, src_mask)
        x = self.norm2_d(x + self.dropout(x_out))

        #Third sub-layer
        x_out = self.ff_d(x)
        x = self.norm3_d(x + self.dropout(x_out))

        return x


