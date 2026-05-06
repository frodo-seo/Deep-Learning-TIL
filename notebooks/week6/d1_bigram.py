"""Week 6 · D1 — Bigram 언어모델: 카운트 기반 → 신경망 기반"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════
# 데이터: 짧은 이름 목록
# ══════════════════════════════════════
names = ["emma", "olivia", "ava", "sophia", "mia", "luna", "ella", "aria"]

# 글자 → 숫자 매핑
chars = sorted(list(set(''.join(names))))
stoi = {c: i+1 for i, c in enumerate(chars)}
stoi['.'] = 0  # 시작/끝 토큰
itos = {i: c for c, i in stoi.items()}
vocab_size = len(stoi)

print(f"어휘: {itos}")
print(f"어휘 크기: {vocab_size}\n")

# ══════════════════════════════════════
# 방법 1: 카운트 기반 (통계)
# ══════════════════════════════════════
print("=== 카운트 기반 bigram ===\n")

# 2D 카운트 행렬: counts[i][j] = 글자 i 다음에 j가 나온 횟수
counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int32)

for name in names:
    name = '.' + name + '.'  # 시작/끝 표시
    for ch1, ch2 in zip(name, name[1:]):
        counts[stoi[ch1], stoi[ch2]] += 1

# 확률로 변환 (각 행의 합 = 1)
probs = counts.float()
probs = probs / probs.sum(dim=1, keepdim=True)

# 생성해보기
print("카운트 기반 생성:")
torch.manual_seed(42)
for _ in range(5):
    out = []
    ix = 0  # '.'에서 시작
    while True:
        p = probs[ix]
        ix = torch.multinomial(p, 1).item()  # 확률에 따라 샘플링
        if ix == 0:  # '.'이면 끝
            break
        out.append(itos[ix])
    print(f"  {''.join(out)}")

# ══════════════════════════════════════
# 방법 2: 신경망 기반 (학습)
# ══════════════════════════════════════
print("\n=== 신경망 기반 bigram ===\n")

# 학습 데이터 만들기: (현재 글자, 다음 글자) 쌍
xs, ys = [], []
for name in names:
    name = '.' + name + '.'
    for ch1, ch2 in zip(name, name[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(f"학습 데이터: {len(xs)}개의 (현재, 다음) 쌍")
print(f"예시: '{itos[xs[0].item()]}' → '{itos[ys[0].item()]}'")
print(f"      '{itos[xs[1].item()]}' → '{itos[ys[1].item()]}'\n")

# 모델: 원핫 → Linear → Softmax → CE
# 사실상 룩업 테이블 학습
W = torch.randn(vocab_size, vocab_size, requires_grad=True)

# 학습
for epoch in range(200):
    # forward
    xenc = F.one_hot(xs, num_classes=vocab_size).float()  # 원핫 인코딩
    logits = xenc @ W                                      # 예측 점수
    loss = F.cross_entropy(logits, ys)                     # softmax + CE

    # backward
    W.grad = None
    loss.backward()

    # update
    W.data -= 0.1 * W.grad

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}")

# 생성해보기
print("\n신경망 기반 생성:")
torch.manual_seed(42)
for _ in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = xenc @ W
        p = F.softmax(logits, dim=1)
        ix = torch.multinomial(p, 1).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(f"  {''.join(out)}")

print("\n두 방법의 결과가 비슷할 것 — 같은 패턴을 학습하니까!")
