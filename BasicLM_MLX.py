# Imports
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx.core import einsum
from einops import rearrange
from tqdm import tqdm
# from tqdm.notebook import tqdm
import tiktoken
import numpy as np

#Global Parameters

class MultiHeadAttention(nn.Module):
    
    def __init__(self, block_size, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.qkv_weights = mx.random.normal((embedding_dim, 3*embedding_dim)) / math.sqrt(embedding_dim)
        self.out_weights = mx.random.normal((embedding_dim, embedding_dim)) / math.sqrt(embedding_dim)

        self._tril = mx.tril(mx.ones((block_size, block_size)))

    
    def __call__(self, x):
        #x.shape is (b,l,e) - b is batch size, l is sequence length, e is embedding vector length
        b,l,e = x.shape

        qkv = einsum('b l e, e d -> b l d', x, self.qkv_weights)
        qkv = rearrange(qkv, 'b l (three n d) -> three b n l d', three=3, n=self.num_heads)
        q,k,v = qkv[0],qkv[1],qkv[2]

        attn = einsum('b n q d, b n k d -> b n q k', q, k)

        attn = attn / math.sqrt(q.shape[-1])

        attn = mx.where(self._tril[:l, :l] == 1, attn, float('-inf'))
        attn = mx.softmax(attn, axis=-1)

        out = einsum('b n q l, b n l d -> b n q d', attn, v)

        out = rearrange(out, 'b n q d -> b q (n d)')
        out = einsum('b l e, e o -> b l o', out, self.out_weights)
        return out


class LayerNorm(nn.Module):

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = mx.ones((d,))
        self.beta = mx.zeros((d,))

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        norm = (x - mean) / mx.sqrt(var + self.eps)
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
    
    def __call__(self, x):
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

    def __call__(self, input_ids, targets=None):
        token_embedding = self.token_embedding_table(input_ids)
        position_embedding = self.position_embedding_table(mx.arange(input_ids.shape[1]))

        x = self.blocks(token_embedding + position_embedding)
        x = self.norm_final(x)
        
        # Instead of output as in the MLS version, while ensuring the weights are tied
        logits = einsum('b l e, v e -> b l v', x, self.token_embedding_table.weight)

        if targets is None:
            loss = None
        else:
            logits = rearrange(logits, 'b l e -> (b l) e')
            targets = rearrange(targets, 'b l -> (b l)')
            loss = nn.losses.cross_entropy(logits, targets, reduction='mean')

        return logits, loss
    
    def generate(self, input_ids, max_new_tokens):
        for _ in tqdm(range(max_new_tokens)):
            input_last_block = input_ids[:, -self.block_size:]
            logits, loss = self(input_last_block)
            logits = logits[:, -1, :]
            next_token = mx.random.categorical(logits, axis=-1)[:, None]
            input_ids = mx.concatenate((input_ids, next_token), axis=1)
            mx.eval(input_ids)
        return input_ids



class ModelWrapper():
    
    def __init__(self, config, text):
        self.config = config

        data = mx.array(np.array(self.config['tokenizer'].encode(text), dtype=np.int32))
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

        num_params = sum(p.size for _, p in tree_flatten(self.model.parameters()))
        print(f"{num_params/1e6:.2f} M parameters")
    
    def batch(self, split, batch_size, block_size):
        if split == 'train':
            data = self.trainset
        elif split == 'val':
            data = self.valset
        inds = np.random.randint(len(data) - block_size, size=batch_size)
        x = mx.stack([data[i:i+block_size] for i in inds])
        y = mx.stack([data[i+1:i+block_size+1] for i in inds])
        return x, y
    

    def estimate_loss(self):
        out = {}
        for split in ['train', 'val']:
            total = 0.0
            for _ in range(self.config['eval_iters']):
                X, Y = self.batch(split, self.config['batch_size'], self.config['block_size'])
                _, loss = self.model(X, Y)
                mx.eval(loss)
                total += loss.item()
            out[split] = total / self.config['eval_iters']
        return out

    def loss_fn(self, model, X, Y):
        _, loss = model(X, Y)
        return loss
    
    def train(self):
        loss_and_grad = nn.value_and_grad(self.model, self.loss_fn)
        optimizer = optim.AdamW(
            learning_rate=self.config['learning_rate'],
            bias_correction=True,
        )

        for it in tqdm(range(self.config['max_iters'])):
            if it % self.config['eval_interval'] == 0 or it == self.config['max_iters'] - 1:
                losses = self.estimate_loss()
                tqdm.write(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")

            xb, yb = self.batch('train', self.config['batch_size'], self.config['block_size'])
            loss, grads = loss_and_grad(self.model, xb, yb)
            optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), optimizer.state)
    
    def generate_text(self, max_new_tokens=2000):
        context = mx.zeros((1, 1), dtype=mx.int32)
        new_tokens = self.model.generate(context, max_new_tokens=max_new_tokens)[0]
        mx.eval(new_tokens)
        print(self.config['tokenizer'].decode(new_tokens.tolist()))  
