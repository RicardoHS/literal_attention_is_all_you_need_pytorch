import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError()


class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError()


class PositionWiseFullyConnectedFeedForward(nn.Module):
    # 3.3 Position-wise Feed-Forward Networks
    # TODO: check this shit
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear_tranformation_one = nn.Linear(d_model, d_ff)
        self.linear_tranformation_two = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # TODO: droput here? check in the googles code
        relu = nn.functional.relu(self.linear_tranformation_one(x))
        return self.linear_tranformation_two(relu)


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
        # x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.shape[0]
        positional_encoder_matrix = self.pe[:,:seq_len]
        x = x + positional_encoder_matrix
        return x