### Libraries ###
import torch
import torch.nn as nn
from mid import Encoder, Decoder
from low import PositionalEncoding

class Transformer(nn.Module):
    """
    Transformer.
    The Transformer model consists of an encoder-decoder architecture using self-attention mechanisms for sequence-to-sequence tasks, such as translation.
    
    Params:
    - dmodel: Number of dimensions for each token.
    - src_vocab_size: Size of the source vocabulary.
    - tgt_vocab_size: Size of the target vocabulary.
    - seq_length: Maximum length of the source sequence.
    - dec_seq_length: Maximum length of the target sequence.
    - num_head: Number of attention heads.
    - num_layers: Number of encoder and decoder layers.
    - dropout: Dropout rate for regularization.
    - device: Device on which the model will run (GPU or CPU).
    - src_pad_idx: Padding index for the source sequence (default: 0).
    - tgt_pad_idx: Padding index for the target sequence (default: 0).
    - tgt_sos_idx: Start of sequence index for the target sequence (default: 0).
    """
    
    def __init__(self, 
                 dmodel, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 seq_length, 
                 dec_seq_length, 
                 num_head,
                 num_layers, 
                 dropout, 
                 device,
                 src_pad_idx = 0, tgt_pad_idx = 0, tgt_sos_idx = 0):
        super(Transformer, self).__init__()
        
        #GPU or CPU
        self.device = device
        
        #Embedding + Positional Encoding
        self.embd_enc = nn.Embedding(src_vocab_size, dmodel)
        self.embd_dec = nn.Embedding(tgt_vocab_size, dmodel)
        self.positional = PositionalEncoding(dmodel, seq_length)

        #Filters
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        
        #Encoder
        self.enc = nn.ModuleList([Encoder(dmodel, num_head, dropout) for _ in range(num_layers)]) 
        #Decoder
        self.dec = nn.ModuleList([Decoder(dmodel, num_head, dropout) for _ in range(num_layers)])

        #Output
        self.linear = nn.Linear(dmodel, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for linear and embedding layers.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.2)

    def make_src_mask(self, src):
        """
        Create a source mask to prevent the model from attending to padding tokens in the source sequence.
        
        Params:
        - src: Input tensor of shape (batch_size, seq_length).
        
        Returns:
        - src_mask: Mask tensor of shape (batch_size, 1, 1, src_seq_length).
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(src.device)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        Create a target mask to prevent the model from attending to future tokens and padding tokens in the target sequence.
        
        Params:
        - tgt: Input tensor of shape (batch_size, dec_seq_length).
        
        Returns:
        - tgt_mask: Mask tensor of shape (batch_size, 1, dec_seq_length, dec_seq_length).
        """
        
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(self.device)
        tgt_len = tgt.size(1)

        tgt_sub_mask = (1 - torch.triu(torch.ones(1, tgt_len, tgt_len), diagonal=1)).bool().to(self.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return tgt_mask

    def forward(self, src, tgt):
         """
        Forward pass of the Transformer model.
        
        The input source and target sequences are passed through the encoder and decoder stacks, followed by a linear layer to generate the final output.

        Params:
        - src: Source input tensor of shape (batch_size, seq_length).
        - tgt: Target input tensor of shape (batch_size, dec_seq_length).

        Returns:
        - out: Output tensor of shape (batch_size, dec_seq_length, tgt_vocab_size).
        """
        assert torch.max(src).item() < self.embd_enc.num_embeddings, "Index in src out of range"
        assert torch.max(tgt).item() < self.embd_dec.num_embeddings, "Index in tgt out of range"
    
        src = src.to(self.embd_enc.weight.device)
        src_mask = self.make_src_mask(src)
        enc_out = self.dropout(self.positional(self.embd_enc(src)).to(self.embd_enc.weight.device))
        
        tgt = tgt.to(self.embd_dec.weight.device)
        tgt_mask = self.make_tgt_mask(tgt).to(self.embd_dec.weight.device)
        dec_out = self.dropout(self.positional(self.embd_dec(tgt)).to(self.embd_dec.weight.device))    
        
        # Pass through the encoder layers
        for enc in self.enc:
            enc_out = enc(enc_out, src_mask)
    
        # Pass through the decoder layers
        for dec in self.dec:
            dec_out = dec(dec_out, enc_out, src_mask, tgt_mask)
    
        out = self.linear(dec_out)
        return out
