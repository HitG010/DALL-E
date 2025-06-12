import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_embeddings):
        super(Encoder, self).__init__()
        
        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
        ])
        
        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=False)
            )
        ])
        
        self.encoder_quant_conv = nn.Sequential(
            nn.Conv2d(64, num_embeddings, kernel_size=1))
        
        
    def forward(self, x):
        out = x
        
        for layer in self.encoder_layers:
            out = layer(out)
            
        for layer in self.residuals:
            out = out + layer(out)
            
        out = self.encoder_quant_conv(out)
        
        return out