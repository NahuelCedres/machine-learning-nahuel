### Libraries ###
import torch
import torch.nn as nn
import math

### Positional Encoding
class PositionalEncoding(nn.Module):
    """
    PositionalEncoding.
    This part of the code handles positional encoding, which assigns a positional value to each token in the sentence.

    Params:
    - dmodel: number of dimension for each token.
    - max_seq_length: max length of the setence.
    """
    def __init__(self, dmodel, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # Base Matrix
        pe = torch.zeros(max_seq_length, dmodel)
        pe.requires_grad = False 

        #Position  value
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1)
        
        #"Similarity" between words based on your position in the text
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() * -(math.log(10000.0) / dmodel))

        pe[:, 0::2] = torch.sin(position * div_term) #even
        pe[:, 1::2] = torch.cos(position * div_term) #odd 

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


### One single self-attention head
class Head(nn.Module):
    """
    One single self-attention head.
    This function calculates the output for a single self-attention head, determining what the head considers important.

    Params:
    - dmodel: number of dimension for each token.
    - d_k: Head size (num_head / dmodel).
    """
    
    def __init__(self, dmodel, d_k):
        super(Head, self).__init__()

        #Query
        self.Q = nn.Linear(dmodel, d_k, bias = False)
        #Key
        self.K = nn.Linear(dmodel, d_k, bias = False)
        #Value
        self.V = nn.Linear(dmodel, d_k, bias = False)

        #d_k
        self.d_k = d_k

    def forward(self, q, k, v, mask = None):
        ### Scaled dot-product attention ###

        #Obtain Q, K-V 
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)

        #MatMul and Scale
        w = q @ k.transpose(-2, -1) * k.shape[-1] / math.sqrt(self.d_k)

        #Apply mosk?
        if mask is not None:
            w = w.masked_fill(mask == 0, float('-inf'))

        #Applies softmax
        w = nn.functional.softmax(w, dim = -1)

        #the last MatMul
        out = w @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention.
    Combines the outputs of multiple attention heads into a single output.
    
    Params:
    - dmodel: Number of dimensions for each token.
    - num_head: Number of attention heads.
    - dropout: Dropout rate for the dropout layer.
    """
    def __init__(self, dmodel, num_head, dropout):
        super(MultiHeadAttention, self).__init__()

        assert dmodel % num_head == 0, "dmodel % num_head must be 0."
        self.d_k = int(dmodel // num_head) # Get d_k param
        
        #ModuleList of heads
        self.heads = nn.ModuleList([Head(dmodel, self.d_k) for _ in range(num_head)])

        #Linear layer to project the concatenated heads
        self.linear = nn.Linear(dmodel, dmodel)
        #Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask = None):
        #Concatenate each output head in one vector.
        concat_heads = torch.cat([h(Q, K, V, mask) for h in self.heads], dim = -1)
        out = self.dropout(self.linear(concat_heads))
        
        return out


class FeedForward(nn.Module):
    """
    FeedForward.
    A key component that takes the output of the self-attention mechanism and transforms it into the final output of the model.
    
    Params:
    - dmodel: Number of dimensions for each token.
    - dropout: Dropout rate for the dropout layer.
    """
    def __init__(self, dmodel, dropout):
        super(FeedForward, self).__init__()

        #Linear
        self.linear1 = nn.Linear(dmodel, dmodel * 4)
        self.linear2 = nn.Linear(dmodel * 4, dmodel)
        
        #ReLU and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        out = self.dropout(self.linear2(x))

        return out



        




        