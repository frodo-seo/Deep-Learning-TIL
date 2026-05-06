"""nanoGPT — 진짜 동작하는 작은 GPT"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """Week 7에서 배운 Multi-Head Attention + Causal Mask"""

    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

        # Causal Mask (미래 못 보게)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class Block(nn.Module):
    """Transformer 블록 = Attention + FFN + Residual + LayerNorm"""

    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # Pre-LN + Attention + Residual
        x = x + self.ffn(self.ln2(x))    # Pre-LN + FFN + Residual
        return x


class GPT(nn.Module):
    """전체 GPT 모델"""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads, max_len) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)                          # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
        x = self.drop(tok + pos)                         # (B, T, d_model)
        x = self.blocks(x)                               # (B, T, d_model)
        x = self.ln_final(x)                             # (B, T, d_model)
        logits = self.head(x)                            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.max_len:]             # 최대 길이 제한
            logits, _ = self(idx_crop)                    # forward
            logits = logits[:, -1, :] / temperature       # 마지막 토큰만
            probs = F.softmax(logits, dim=-1)             # 확률
            next_token = torch.multinomial(probs, 1)      # 샘플링
            idx = torch.cat([idx, next_token], dim=1)     # 이어붙이기
        return idx
