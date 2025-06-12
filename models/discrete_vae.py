import torch
import torch.nn as nn

class DiscreteVAE(nn.Module):
    def __init__(self, num_embeddings = 1024, embedding_dim = 512):
        super(DiscreteVAE, self).__init__()
        
        # self.encoder = Encoder(num_embeddings = )