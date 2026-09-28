"""
mini_gpt.py

Assignment 2: Building a Small-Scale Foundation Model from Scratch
====================================================================

A minimal, from-scratch GPT-style decoder-only transformer implemented in
PyTorch, satisfying the assignment's model requirements:

  - 1-2 transformer blocks (`n_layer`)
  - Embedding dimension 64-256 (`n_embd`)
  - Multi-head causal self-attention with 2-4 heads (`n_head`)
  - Learned positional embeddings
  - Layer normalization (pre-norm) around both the attention and MLP
    sub-blocks
  - GELU activation in the MLP
  - A final linear head projecting to vocabulary-sized logits for
    next-token prediction

The attention + masking logic here was first validated independently in
`dev_validate_numpy.py` (a pure-numpy, no-autograd re-implementation of the
same shapes and causal-masking behavior) before being written here, so
that any bug in the reshape/transpose/mask arithmetic would be caught
before relying on PyTorch's autograd to "just work".
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniGPTConfig:
    vocab_size: int = 50257     # matches the GPT-2 tokenizer used in Assignment 1
    block_size: int = 64        # sequence length per training example (assignment: 32-128)
    n_layer: int = 2            # number of transformer blocks (assignment: 1-2)
    n_head: int = 4             # number of attention heads (assignment: 2-4)
    n_embd: int = 128           # embedding dimension (assignment: 64-256)
    dropout: float = 0.1
    pad_token_id: int = 50256   # GPT-2 has no dedicated pad token; Assignment 1 used eos_token as pad


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention restricted to the causal (left-to-right)
    direction: position i may only attend to positions <= i. A key-padding
    mask is also supported so that attention never attends to pad tokens
    introduced by the DataLoader's collate function in Assignment 1.
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular causal mask, precomputed once for the max sequence
        # length this model will ever see (`block_size`).
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))

        if key_padding_mask is not None:
            # key_padding_mask: (B, T) with 1 = real token, 0 = pad token.
            # Broadcast to (B, 1, 1, T) so it masks out padded *key*
            # positions for every query position and every head.
            pad_mask = key_padding_mask[:, None, None, :].bool()
            att = att.masked_fill(~pad_mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network with a 4x expansion, as in the original GPT/Transformer design."""

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """One transformer block: pre-norm causal self-attention + pre-norm MLP, both with residual connections."""

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    A minimal GPT-style decoder-only transformer for next-token prediction.

    forward(idx, attention_mask, targets) returns (logits, loss). `loss` is
    None if `targets` is not provided (e.g. at generation time).
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(self, idx: torch.Tensor, attention_mask: torch.Tensor = None, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds this model's block_size {self.config.block_size}"
        )

        tok = self.tok_emb(idx)                      # (B, T, C)
        pos = self.pos_emb[:, :T, :]                  # (1, T, C), broadcasts over batch
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x, key_padding_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # positions to skip (e.g. padding in the target)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """Simple autoregressive sampling loop, mainly useful as a sanity check after training."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
