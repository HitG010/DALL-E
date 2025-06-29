import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)

class DallEGPTConfig:
    def __init__(self, text_vocab_size, image_vocab_size, max_sequence_len, im_size, **kwargs):
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        self.block_size = max_sequence_len
        self.im_size = im_size
        self.num_text_tokens = max_sequence_len - (im_size * im_size)
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
            
class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).(1, 1, config.block_size, config.block_size))
        
        self.n_head = config.n_head
        
        
    def forward(self, x, layer_past = None):
        B, T, C = x.size()
        
        k = self.key(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        
        y = self.resid_drop(self.proj(y))
        return y
    
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
    
    class GPT(nn.Module):
        
        def __init__(self, config):
            super().__init__()
            
            self.text_tok_emb = nn.Embedding(config.text_vocab_size, config.n_embd)
            self.image_tok_emb = nn.Embedding(config.image_vocab_size, config.n_embd)
            
            self.text_pos_emb = nn.Parameter(torch.zeros(1, config.num_text_tokens, config.n_embd))
            self.image_pos_emb = nn.Parameter(torch.zeros(1, config.im_size * config.im_size, config.n_embd))
            
            self.drop = nn.Dropout(config.embd_pdrop)
            
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
            
            self.ln_f = nn.LayerNorm(config.n_embd)
            self.head = nn.Linear(config.n_embd, config.text_vocab_size + config.image_vocab_size, bias=False)
            self.config = config
            self.block_size = config.block_size
            self.apply(self.init_weights)
            
            logger.info("GPT model initialized with config: %s", config.__dict__)
            
        def get_block_size(self):
            return self.block_size
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isInstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
                
            elif isinstance(module, GPT):
                nn.init.normal_(module.text_pos_emb, mean=0.0, std=0.02)
                nn.init.normal_(module.image_pos_emb, mean=0.0, std=0.02)
                
                
        def forward(self, image_tokens, text_tokens, targets = None,):
            b, im_t = image_tokens.size()
            b, tx_t = text_tokens.size()
            
            assert im_t + tx_t <= self.block_size, "Input sequence length exceeds block size"
            
            text_emb = self.text_tok_emb(text_tokens)
            text_pos = self.text_pos_emb[:, :tx_t, :]
            text_token_embeddings = self.drop(text_emb + text_pos)
            x = text_token_embeddings
            
            if im_t > 0:
                image_emb = self.image_tok_emb(image_tokens)
                image_pos = self.image_pos_emb[:, :im_t, :]
                image_token_embeddings = self.drop(image_emb + image_pos)
                x = torch.cat((image_token_embeddings, text_token_embeddings), dim=1)
                
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.head(x)
            
            loss_text = None
            loss_image = None
            
            if targets is not None:
                logits = logits[:, :-1, :]
                
                text_logits = logits[:, :text_t - 1, :].permute(0, 2, 1)
                image_logits = logits[:, text_t - 1:, :].permute(0, 2, 1)
                
                text_logits[:, self.config.text_vocab_size:, :] = -torch.finfo(logits.dtype).max
                image_logits[:, :self.config.text_vocab_size, :] = -torch.finfo(logits.dtype).max
                
                loss_text = F.cross_entropy(text_logits, targets[:, :text_t-1])
                loss_image = F.cross_entropy(image_logits, targets[:, text_t-1:])
            
            return logits, loss_text, loss_image
           