import torch
import torch.nn as nn


class Decoder(nn.Module):
    
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        
        self.decoder_layers = nn.ModuleList([
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ])
        
        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU()
            )
        ])
        
        self.decoder_quant_conv = nn.Sequential(nn.Conv2d(embedding_dim, 64, kernel_size=1))
        
        
    def forward(self, x):
        out = self.decoder_quant_conv(x)
        
        for layer in self.residuals:
            out += layer(out)
            
        for layer in self.decoder_layers:
            out = layer(out)
            
        return out