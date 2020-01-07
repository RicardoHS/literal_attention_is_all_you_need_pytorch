import torch
import torch.nn as nn
import math

import matplotlib.pyplot as plt
import seaborn as sns

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        # sns.heatmap(self.pe.view(80,512).numpy(), cmap=sns.color_palette("RdBu_r", 80));plt.xlabel('i');plt.ylabel('pos');plt.show()
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.shape[0]
        positional_encoder_matrix = self.pe[:,:seq_len]
        x = x + positional_encoder_matrix
        return x


kk = PositionalEncoder(d_model=512)

a = torch.Tensor(80*512).view(80,512)
b = kk(a)

print(a)
print(b)
assert torch.all(torch.eq(torch.round(b), torch.round(b_checked)))