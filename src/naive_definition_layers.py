import torch
import torch.nn as nn


class Attention(nn.Module):
    # 3.2 - Attention
    def __init__(self):
        super().__init__()

    def scaled_dot_product_attention(self, query, key, value):
        # 3.2.1 - Scaled Dot-Product Attention
        # query and key has dimension d_k
        # value has dimension d_v
        d_k = query.shape[0] #TODO: Check this shape

        # 3.2 - Attention - Compatibility function
        dot_product = torch.matmul(query, key.transpose())
        scaled_dot_product = dot_product/torch.sqrt(d_k)
        # 3.2.3 Applications of Attention in our Model
        # Similarly, self-attention layers in the decoder...
        # TODO: define mask
        mask = 0

        softmax = nn.functional.softmax(self.scaled_dot_product_attention(query,key))
        return nn.matmul(softmax, value)

    def forward(self, query, key, value):
        return self.scaled_dot_product_attention(query, key, value)

class MultiHeadAttention(nn.Module):
    # 3.2.2 - Multi-Head Attention
    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.WQ = [nn.Linear(d_model, d_k) for _ in range(0, h)]
        self.WK = [nn.Linear(d_model, d_k) for _ in range(0, h)]
        self.WV = [nn.Linear(d_model, d_v) for _ in range(0, h)]
        self.WO = nn.Linear(h*d_v, d_model)

        self.h = h

    def forward(self, query, key, value):
        # TODO: Paralelizate this (in the paper it is stated that this is done in parallel)
        def compute_head(self, i):
            q = nn.matmul(query, self.WQ[i])
            k = nn.matmul(key, self.WK[i])
            v = nn.matmul(value, self.WV[i])
            return Attention().forward(q,k,v)

        head_i = [compute_head(i) for i in range(0, self.h)]

        # TODO: maybe is torch.stack instead of torch.cat
        return nn.matmul(torch.cat(head_i), self.WO) 