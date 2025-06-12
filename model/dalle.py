import torch
import torch.nn as nn
from models.mingpt import GPT, DallEGPTCofig

class DallE(nn.Module):
    
    def __init__(self, vae, num_words, image_size, max_text_len, image_vocab_size, gpt_config):
        super(DallE, self).__init__()
        
        self.num_words = num_words
        self.image_size = image_size
        self.max_text_len = max_text_len
        
        self.image_vocab_size = image_vocab_size
        
        max_sequence_len = max_text_len + (image_size * image_size)
        
        config = DallEGPTCofig(
            text_vocab_size=num_words,
            image_vocab_size=image_vocab_size,
            max_sequence_len=max_sequence_len,
            im_size=image_size,
            **gpt_config
        )
        self.gpt = GPT(config)
        
    def forward(self, im, text):
        
        image_tokens = self.vae.get_codebook_indices(im).reshape(im.size(0), -1)
        
        target_image_tokens = image_tokens + self.num_words
        
        labels = None
        
        if self.training:
            labels = torch.cat((text[:, 1:], target_image_tokens), dim=1)
        
        
        logits, loss_text, loss_image = self.gpt(image_tokens, text, targets=labels)
        return logits, loss_text, loss_image