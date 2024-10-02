import os
import h5py
import math


import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoderLayer, TransformerEncoder

# TRIED BUT IT TOOK TOO LONG JUST TO LOAD THE DATA TO GPU
# """
# 145 base pairs
# 12 targets
#     0  "k562_minp_rep1": K562 cell line, minimal promoter, replicate 1
#     1  "k562_minp_rep2": K562 cell line, minimal promoter, replicate 2
#     2  "k562_minp_avg": K562 cell line, minimal promoter, average*
#     3  "k562_sv40p_rep1": K562 cell line, strong SV40 promoter, replicate 1
#     4  "k562_sv40p_rep1": K562 cell line, strong SV40 promoter, replicate 2
#     5  "k562_sv40p_avg": K562 cell line, strong SV40 promoter, average*
#     6  "hepg2_minp_rep1": HepG2 cell line, minimal promoter, replicate 1
#     7  "hepg2_minp_rep2": HepG2 cell line, minimal promoter, replicate 2
#     8  "hepg2_minp_avg": HepG2 cell line, minimal promoter, average*
#     9  "hepg2_sv40p_rep1": HepG2 cell line, strong SV40 promoter, replicate 1
#     10 "hepg2_sv40p_rep1": HepG2 cell line, strong SV40 promoter, replicate 2
#     11 "hepg2_sv40p_rep1": HepG2 cell line, strong SV40 promoter, average*
# """

# class MPRADataLoader(Dataset):
#     def __init__(self, data_folder, datatype):
#         self.batch_size = 128
#         self.fname = os.path.join(data_folder, datatype + '.hdf5')

#         with h5py.File(self.fname, 'r') as hf:
#             self.max_batches = hf['X']['sequence'].shape[0] // self.batch_size
#             self.x_shape = hf['X']['sequence'].shape
#             self.y_shape = hf['Y']['output'].shape
        

#     def __len__(self):
#         return self.max_batches * self.batch_size

#     def __getitem__(self, idx):
#         with h5py.File(self.fname, 'r') as hf:
#             x = hf['X']['sequence'][idx]
#             y = hf['Y']['output'][idx]
        
#         # Optionally: Convert to torch tensors
#         x = torch.tensor(x)
#         y = torch.tensor(y)
        
#         return x, y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[num_tokens, batch_size, embedding_dim]``
        """

        seq_len = x.size(0)
        return self.dropout(x + self.pe[:seq_len])
    
class TransformerModel(nn.Module):
    def __init__(self, input_len = 170, d_model = 64, output_dim=1):
        super(TransformerModel, self).__init__()
        self.input_len = input_len
        self.d_model = d_model
        self.output_dim = output_dim

        # extract features with CNN
        self.feature_extraction = nn.Sequential(
                                    nn.Conv1d(4, 16, kernel_size=13, padding=6),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv1d(16, 32, kernel_size=7, padding=3),
                                    nn.LeakyReLU(0.2),
                                )
        
        # Linear Embedding; adding some variations such as dropout and activations
        self.embedding = nn.Sequential(
            nn.Linear(32, self.d_model),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
        )

        # positional encoding for Transformers
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, dropout=0.2, max_len = self.input_len)
        
        self.CLS_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Transformers 
        encoder_layers = TransformerEncoderLayer(self.d_model, 8, 160, 0.2)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 4)
        self.linear_layer_after_transformer = nn.Sequential(nn.Linear(self.d_model, self.d_model // 2),
                                               nn.LeakyReLU(0.2))
        # final layers
        self.final_layers = nn.Sequential(nn.Conv1d(1, 32, kernel_size=3, padding=1),
                                          nn.LeakyReLU(0.2),
                                          nn.Conv1d(32, 16, kernel_size=3, padding=1),
                                          nn.LeakyReLU(0.2),
                                          nn.MaxPool1d(2,2))
        
        self.final_layer_linear= nn.Sequential(nn.Linear(16 * (self.d_model // 4), 128),
                                                nn.BatchNorm1d(128),
                                                nn.LeakyReLU(0.2),
                                                nn.Linear(128, 64),
                                                nn.LeakyReLU(0.2),
                                                nn.Linear(64, 32),
                                                nn.LeakyReLU(0.2),
                                                nn.Linear(32, self.output_dim))


    def forward(self, x):
        BS, L, F = x.shape

        x = self.feature_extraction(x).permute(0,2,1)

        # x: BS, BPWF, F (BS, 68, 32)
        # BPWF: base pair windows features
        # F: Features 
        x = self.embedding(x)

        x = self.pos_encoder(x.permute(1,0,2))

        # classification tokens
        CLS_token = self.CLS_token.expand(-1, BS, -1)
        x = torch.cat([CLS_token.to(x.device), x], axis=0)

        # take only classifications tokens
        x = self.transformer_encoder(x)[0,:,:].unsqueeze(1)
        
        x = self.linear_layer_after_transformer(x)
        x = self.final_layers(x).view(BS, 16 * (self.d_model // 4))
        x = self.final_layer_linear(x)
        
        return x
    
# # # Sample runner for your model
# def sample_runner():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = TransformerModel().to(device)

#     sample_input = torch.randn(2, 4, 170).to(device) 
#     output = model(sample_input)

#     print(f"Output shape: {output.shape}")  # Should be (BS, 12)
#     print(output)

# # Run the sample
# sample_runner()