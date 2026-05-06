"""Week 6 · D3 — MLP 언어모델: 문맥 여러 글자 보기"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 데이터 ──
names = ["emma", "olivia", "ava", "sophia", "mia", "luna", "ella", "aria",
         "isabella", "emily", "amelia", "harper", "evelyn", "abigail"]

chars = sorted(list(set(''.join(names))))
stoi = {c: i+1 for i, c in enumerate(chars)}
stoi['.'] = 0
itos = {i: c for c, i in stoi.items()}
vocab_size = len(stoi)

# ── 핵심: 문맥 길이 = 3 (이전 3글자를 보고 다음 1글자 예측) ──
context_len = 3

xs, ys = [], []
for name in names:
    context = [0] * context_len  # '...' 으로 시작
    for ch in name + '.':
        xs.append(context[:])
        ys.append(stoi[ch])
        context = context[1:] + [stoi[ch]]  # 윈도우 슬라이드

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(f"학습 데이터: {len(xs)}개")
print(f"예시: {[itos[i] for i in xs[0].tolist()]} → '{itos[ys[0].item()]}'")
print(f"      {[itos[i] for i in xs[1].tolist()]} → '{itos[ys[1].item()]}'")
print(f"      {[itos[i] for i in xs[2].tolist()]} → '{itos[ys[2].item()]}'")
print()

# ── 모델 ──
emb_dim = 10     # 각 글자를 10차원 벡터로
hidden = 64      # 은닉층 뉴런 수

torch.manual_seed(42)
C = nn.Embedding(vocab_size, emb_dim)        # 임베딩 테이블
W1 = nn.Linear(context_len * emb_dim, hidden) # 문맥 → 은닉층
W2 = nn.Linear(hidden, vocab_size)            # 은닉층 → 예측

params = list(C.parameters()) + list(W1.parameters()) + list(W2.parameters())
print(f"총 파라미터: {sum(p.numel() for p in params)}개\n")

# ── 학습 ──
optimizer = torch.optim.Adam(params, lr=0.01)

for epoch in range(500):
    # forward
    emb = C(xs)                          # (N, 3, 10) — 각 글자를 임베딩
    emb_flat = emb.view(-1, context_len * emb_dim)  # (N, 30) — 3개 이어붙이기
    h = torch.tanh(W1(emb_flat))         # (N, 64) — 은닉층
    logits = W2(h)                       # (N, vocab_size) — 예측 점수
    loss = F.cross_entropy(logits, ys)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}")

# ── 생성 ──
print("\n생성 결과:")
torch.manual_seed(42)
for _ in range(8):
    out = []
    context = [0] * context_len
    while True:
        emb = C(torch.tensor([context]))
        emb_flat = emb.view(1, -1)
        h = torch.tanh(W1(emb_flat))
        logits = W2(h)
        p = F.softmax(logits, dim=1)
        ix = torch.multinomial(p, 1).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    print(f"  {''.join(out)}")
