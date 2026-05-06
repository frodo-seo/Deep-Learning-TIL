"""Week 8 — GPT 구조 전체를 눈으로 보기"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ── 설정 ──
vocab_size = 20      # 어휘 크기 (작게)
d_model = 64         # 임베딩 차원
n_heads = 4          # Attention Head 수
d_ff = 256           # FFN 중간 차원 (4배)
n_layers = 2         # Transformer 블록 수
max_len = 16         # 최대 시퀀스 길이

print("=" * 60)
print("GPT 구조 시각화")
print("=" * 60)

# ── 입력: 토큰 3개 ──
tokens = torch.tensor([[2, 7, 13]])   # 배치 1, 토큰 3개
B, T = tokens.shape
print(f"\n[입력] 토큰: {tokens.tolist()}")
print(f"  shape: ({B}, {T})")

# ══════════════════════════════════════
# 1단계: 임베딩
# ══════════════════════════════════════
tok_emb = nn.Embedding(vocab_size, d_model)
pos_emb = nn.Embedding(max_len, d_model)

positions = torch.arange(T)   # [0, 1, 2]
x = tok_emb(tokens) + pos_emb(positions)

print(f"\n[1단계: 임베딩]")
print(f"  토큰 임베딩: ({B}, {T}) → ({B}, {T}, {d_model})")
print(f"  위치 임베딩: ({T},) → ({T}, {d_model})")
print(f"  합산 결과 x: {x.shape}")

# ══════════════════════════════════════
# 2단계: Transformer 블록
# ══════════════════════════════════════
for layer_idx in range(n_layers):
    print(f"\n{'─' * 60}")
    print(f"[2단계: Transformer 블록 {layer_idx + 1}]")

    # ── LayerNorm ──
    ln1 = nn.LayerNorm(d_model)
    x_norm = ln1(x)
    print(f"\n  LayerNorm: {x_norm.shape}")

    # ── Multi-Head Attention ──
    W_q = nn.Linear(d_model, d_model, bias=False)
    W_k = nn.Linear(d_model, d_model, bias=False)
    W_v = nn.Linear(d_model, d_model, bias=False)

    Q = W_q(x_norm)  # (1, 3, 64)
    K = W_k(x_norm)
    V = W_v(x_norm)
    print(f"\n  Q, K, V 생성:")
    print(f"    x @ W_q → Q: {Q.shape}")
    print(f"    x @ W_k → K: {K.shape}")
    print(f"    x @ W_v → V: {V.shape}")

    # Head로 쪼개기
    head_dim = d_model // n_heads
    Q = Q.view(B, T, n_heads, head_dim).transpose(1, 2)  # (1, 4, 3, 16)
    K = K.view(B, T, n_heads, head_dim).transpose(1, 2)
    V = V.view(B, T, n_heads, head_dim).transpose(1, 2)
    print(f"\n  {n_heads}개 Head로 쪼개기:")
    print(f"    Q: {Q.shape}  ← (배치, 헤드, 토큰, 헤드차원)")
    print(f"    K: {K.shape}")
    print(f"    V: {V.shape}")

    # Q @ K^T
    scores = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)
    print(f"\n  Q @ K^T / √{head_dim}:")
    print(f"    scores: {scores.shape}  ← (배치, 헤드, 토큰, 토큰)")

    # Causal Mask
    mask = torch.tril(torch.ones(T, T))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    print(f"\n  Causal Mask 적용:")
    print(f"    Head 1의 점수 행렬:")
    for i in range(T):
        row = scores[0, 0, i].tolist()
        row_str = [f"{v:6.2f}" if v > -1000 else "  -inf" for v in row]
        print(f"      토큰{i}: [{', '.join(row_str)}]")

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    print(f"\n  Softmax 후 (Head 1의 가중치):")
    for i in range(T):
        row = attn_weights[0, 0, i].tolist()
        row_str = [f"{v:.3f}" for v in row]
        print(f"      토큰{i}: [{', '.join(row_str)}]")

    # @ V
    attn_out = attn_weights @ V
    print(f"\n  가중치 @ V:")
    print(f"    Attention 출력: {attn_out.shape}")

    # Head 합치기
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d_model)
    print(f"\n  Head 이어붙이기:")
    print(f"    ({n_heads} × {head_dim}) = {d_model}차원 → {attn_out.shape}")

    # Residual
    x = x + attn_out
    print(f"\n  + 잔차(Residual): {x.shape}")

    # ── FFN ──
    ln2 = nn.LayerNorm(d_model)
    x_norm2 = ln2(x)
    print(f"\n  LayerNorm: {x_norm2.shape}")

    ff1 = nn.Linear(d_model, d_ff)
    ff2 = nn.Linear(d_ff, d_model)

    ffn_out = ff2(F.relu(ff1(x_norm2)))
    print(f"\n  FFN:")
    print(f"    Linear: ({d_model}) → ({d_ff})  ← 4배 확장")
    print(f"    ReLU")
    print(f"    Linear: ({d_ff}) → ({d_model})  ← 원래 크기")
    print(f"    FFN 출력: {ffn_out.shape}")

    x = x + ffn_out
    print(f"\n  + 잔차(Residual): {x.shape}")

# ══════════════════════════════════════
# 3단계: 최종 예측
# ══════════════════════════════════════
print(f"\n{'─' * 60}")
print(f"[3단계: 최종 예측]")

ln_final = nn.LayerNorm(d_model)
x = ln_final(x)
print(f"\n  최종 LayerNorm: {x.shape}")

last_token = x[:, -1, :]   # 마지막 토큰만
print(f"  마지막 토큰 꺼냄: {last_token.shape}")

head = nn.Linear(d_model, vocab_size)
logits = head(last_token)
print(f"  Linear → logits: {logits.shape}")

probs = F.softmax(logits, dim=-1)
next_token = torch.argmax(probs, dim=-1)
print(f"  softmax → 확률 분포")
print(f"  예측된 다음 토큰: {next_token.item()}")

print(f"\n{'=' * 60}")
print(f"전체 흐름 요약:")
print(f"  토큰 {tokens.tolist()}")
print(f"  → 임베딩 ({B},{T}) → ({B},{T},{d_model})")
print(f"  → Transformer 블록 ×{n_layers} → ({B},{T},{d_model})")
print(f"  → 마지막 토큰 → ({B},{d_model})")
print(f"  → Linear → ({B},{vocab_size})")
print(f"  → softmax → 다음 토큰: {next_token.item()}")
print(f"{'=' * 60}")
