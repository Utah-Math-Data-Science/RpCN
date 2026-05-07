import torch 
import torch.nn as nn
import numpy as np


def build_mlp(input_dim, output_dim, hidden_dim, n_hidden):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    for _ in range(n_hidden - 1): 
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class MeanFlowModel(torch.nn.Module): 
    def __init__(self, input_dim, output_dim, dim, n_hidden, time_embed_dim=32):     
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.net = build_mlp(input_dim + 2 * time_embed_dim, output_dim, dim, n_hidden)
        self.input_dim = input_dim
        

    def time_embedding(self, t):
        """ sinusoidal time embedding """
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        
        half_dim = self.time_embed_dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -embeddings)
        embeddings = t * embeddings.unsqueeze(0)
        
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.time_embed_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
            
        return embeddings
    

    def forward(self, x, t, r): 

        # Get time embeddings
        t_embed = self.time_embedding(t)
        r_embed = self.time_embedding(r)

        # Concatenate along feature dimension
        input_tensor = torch.cat([x, t_embed, r_embed], dim=-1)
        return self.net(input_tensor)
    
