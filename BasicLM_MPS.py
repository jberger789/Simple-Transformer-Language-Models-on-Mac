# Imports
import math, torch
from einops import rearrange
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
# from tqdm import tqdm
from tqdm import tqdm
import tiktoken
import numpy as np

#Global Parameters
device = 'mps' if torch.mps.is_available() else 'cpu'

class MultiHeadAttention(nn.Module):
    
    def __init__(self, block_size, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.qkv_weights = nn.Parameter(torch.randn(embedding_dim, 3*embedding_dim) / math.sqrt(embedding_dim))
        self.out_weights = nn.Parameter(torch.randn(embedding_dim, embedding_dim) / math.sqrt(embedding_dim))

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    
    def forward(self, x):
        #x.shape is (b,l,e) - b is batch size, l is sequence length, e is embedding vector length
        b,l,e = x.shape

        qkv = torch.einsum('b l e, e d -> b l d', x, self.qkv_weights)
        q,k,v = rearrange(qkv, 'b l (three n d) -> three b n l d', three=3, n=self.num_heads).unbind(0)

        attn = einsum('b n q d, b n k d -> b n q k', q, k)

        attn = attn / math.sqrt(q.shape[-1])

        attn = attn.masked_fill(self.tril[:l, :l] == 0, float('-inf'))
        attn = attn.softmax(dim=-1)

        out = einsum('b n q l, b n l d -> b n q d', attn, v)

        out = rearrange(out, 'b n q d -> b q (n d)')
        out = einsum('b l e, e o -> b l o', out, self.out_weights)
        return out


class LayerNorm(nn.Module):

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return (self.gamma * norm) + self.beta

class TransformerBlock(nn.Module):

    def __init__(self, block_size, embedding_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(block_size, embedding_dim, num_heads)
        self.norm_attn = LayerNorm(embedding_dim)
        self.norm_ff = LayerNorm(embedding_dim)

        ff_dim = embedding_dim * 4

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
    
    def forward(self, x):
        attn_out = self.norm_attn(self.attn(x) + x)

        ff_out = self.norm_ff(self.ff(attn_out) + attn_out)

        return ff_out


class BasicLanguageModel(nn.Module):
    def __init__(self, n_vocab, block_size, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(n_vocab, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(block_size, embedding_dim, num_heads) for _ in range(num_layers)])
        self.norm_final = LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, n_vocab)
        self.output.weight = self.token_embedding_table.weight

    def forward(self, input_ids, targets=None):
        token_embedding = self.token_embedding_table(input_ids)
        position_embedding = self.position_embedding_table(torch.arange(input_ids.shape[1], device=device))

        x = self.blocks(token_embedding + position_embedding)
        x = self.norm_final(x)

        logits = self.output(x)

        if targets is None:
            loss = None
        else:
            logits = rearrange(logits, 'b l e -> (b l) e')
            targets = rearrange(targets, 'b l -> (b l)')
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, input_ids, max_new_tokens):
        for _ in tqdm(range(max_new_tokens)):
            input_last_block = input_ids[:, -self.block_size:]
            logits, loss = self(input_last_block)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids



class ModelWrapper():
    
    def __init__(self, config, text):
        self.config = config

        data = torch.from_numpy(np.array(self.config['tokenizer'].encode(text), dtype=np.int32))
        split_index = int(self.config['split_ratio']*len(data))
        self.trainset = data[:split_index]
        self.valset = data[split_index:]

        self.model = BasicLanguageModel(
            n_vocab = self.config['tokenizer'].n_vocab,
            block_size = self.config['block_size'],
            embedding_dim=self.config['embedding_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
        )

        self.m = self.model.to(device)

        print(sum(p.numel() for p in self.m.parameters())/1e6, 'M parameters')
    
    def batch(self, split, batch_size, block_size):
        if split == 'train':
            data = self.trainset
        elif split == 'val':
            data = self.valset
        inds = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in inds])
        y = torch.stack([data[i+1:i+block_size+1] for i in inds])
        x,y = x.to(device), y.to(device)
        return x, y
    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config['eval_iters'])
            for k in tqdm(range(self.config['eval_iters']),leave=False, desc='Estimating loss'):
                X, Y = self.batch(split, self.config['batch_size'], self.config['block_size'])
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])

        for iter in tqdm(range(self.config['max_iters'])):
            # every once in a while evaluate the loss on train and val sets
            if iter % self.config['eval_interval'] == 0 or iter == self.config['max_iters'] - 1:
                losses = self.estimate_loss()
                tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = self.batch('train', self.config['batch_size'], self.config['block_size'])

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    
    def generate_text(self, max_new_tokens=2000):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(self.config['tokenizer'].decode(self.m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))   