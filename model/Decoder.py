import torch
import torch.nn as nn


class Decoder(nn.Module):
    r"""
    Decoder with couple of residual blocks
    followed by conv transpose relu layers
    """
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()
        
        self.decoder_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),  # 👈 reduce to 16
            nn.ReLU(inplace=False),

            nn.Conv2d(16, 3, 1),
            nn.Tanh()
        ])

        
        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=False)),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=False))
        ])
        
        self.decoder_quant_conv = nn.Conv2d(embedding_dim, 64, 1)
        
    
    def forward(self, x):
        out = self.decoder_quant_conv(x)
        for layer in self.residuals:
            out = layer(out) + out
        for idx, layer in enumerate(self.decoder_layers):
            out = layer(out)
            # print(out)
        return out
        


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import yaml
    decoder = Decoder()
    
    out = decoder(torch.rand((3, 64, 14, 14)))
    print(out.shape)
    