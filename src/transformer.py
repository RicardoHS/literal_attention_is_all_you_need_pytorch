import torch
import torch.nn as nn

import sublayers

class Transformer(nn.Module):
    '''
    seq2seq model based entirely in attention mechanism

    Comments in the code references directly the corresponding section in the paper. If the comment
    is on the class definition it means that the main structure of the class is defined in that section.
    If the comment is on the middle of the code it means that specific piece of code is referenced in
    that specific section of the paper.

    arXiv: https://arxiv.org/pdf/1706.03762.pdf

    Usefull lectures:
     - http://jalammar.github.io/illustrated-transformer/
     - http://nlp.seas.harvard.edu/2018/04/03/attention.html
     - https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
     - https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/
     - https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

     TODO:
     - Apply LabelSmoothing 
    '''
    # 3 - Model Architecture
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # 3.5 - Positional Encoding
        self.positional_encoding = sublayers.PositionalEncoder(d_model)

        # 3.4 - Embeddings and Softmax
        # TODO: multiply by sqrt(d_model) (3.4)
        self.input_embedding = nn.Embedding(words_in_vocab, dimensional_embedding)
        self.output_embedding = self.input_embedding
        self.linear = nn.Linear(d_model, d_model) #TODO: Check this param sizes (3.1-encoder and 3.4)

        self.d_k = d_k

    def forward(self, x):
        # 3.4 - Embeddings and Softmax
        encoder_input = self.input_embedding(x) * torch.sqrt(self.d_k) + self.positional_encoding(x)
        z = self.encoder(encoder_input)

        decoder_input = self.input_embedding(previous_transformer_output) * torch.sqrt(self.d_k) + self.positional_encoding(previous_transformer_output)
        y = self.decoder(x, z)
        # 3.4 Embedings and Softmax TODO: Check 3.4 entirely because it makes no sense
        return nn.functional.softmax(self.linear(y))


class Encoder(nn.Module):
    # 3.1 - Encoder and Decoder Stacks
    # Encoder
    def __init__(self):
        super().__init__()        
        self.first_sublayer = sublayers.MultiHeadSelfAttention()
        self.second_sublayer = sublayers.PositionWiseFullyConnectedFeedForward()
        
        self.sublayer_norm1 = nn.LayerNorm()
        self.sublayer_norm2 = nn.LayerNorm()

        # 5.4 - Regularization TODO: Should be there multiples dropout? check in googles code
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        first_sublayer_residual = x
        # 3.2.3 Applications of Attention in our Model
        # The encoder contains self-attention layers. In a self-attention layer all the keys... 
        # TODO: wtf this means?
        # 5.4 - Regularization
        first_sublayer_output = self.sublayer_norm1(first_sublayer_residual + self.dropout(self.first_sublayer(x)))

        second_sublayer_residual = first_sublayer_output
        # 5.4 - Regularization
        second_sublayer_output = self.sublayer_norm2(second_sublayer_residual + self.dropout(self.second_sublayer(x)))

        return second_sublayer_output

class Decoder(nn.Module):
    # 3.1 - Encoder and Decoder Stacks
    # Decoder
    def __init__(self):
        super().__init__()
        self.first_sublayer = sublayers.MaskedMultiHeadSelfAttention()
        self.second_sublayer = sublayers.MultiHeadAttention()
        self.third_sublayer = sublayers.PositionWiseFullyConnectedFeedForward()
        
        self.sublayer_norm1 = nn.LayerNorm()
        self.sublayer_norm2 = nn.LayerNorm()
        self.sublayer_norm3 = nn.LayerNorm()

        # 5.4 - Regularization
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, z):
        # param z: Is the output of the encoder (Section 3)
        # TODO?: Offset the input
        first_sublayer_residual = x
        # 3.2.3 Applications of Attention in our Model
        # Similarly, self-attention layers in the decoder...
        # TODO: wtf is this?
        # 5.4 - Regularization
        first_sublayer_output = self.sublayer_norm1(first_sublayer_residual + self.dropout(self.first_sublayer(x)))

        second_sublayer_residual = first_sublayer_output
        # 3.2.3 Applications of Attention in our Model
        # In "encoder-decoder attention" layers...
        query = first_sublayer_output
        key = z
        value = z 
        # 5.4 - Regularization
        second_sublayer_output = self.sublayer_norm2(second_sublayer_residual + self.dropout(self.second_sublayer(query ,key, value)))

        third_sublayer_residual = second_sublayer_output
        # 5.4 - Regularization
        third_sublayer_output = self.sublayer_norm3(third_sublayer_residual + self.dropout(self.third_sublayer(x)))
