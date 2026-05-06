"""nanoGPT 학습 — Tiny Shakespeare"""

import torch
from model import GPT

# ── 설정 ──
d_model = 128
n_heads = 4
n_layers = 4
max_len = 64
batch_size = 32
lr = 3e-4
max_steps = 3000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ── 데이터 로드 ──
# Tiny Shakespeare 다운로드
import os
data_path = "shakespeare.txt"
if not os.path.exists(data_path):
    print("Shakespeare 데이터 다운로드 중...")
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, data_path)

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"데이터 크기: {len(text):,} 글자")
print(f"처음 100글자: {text[:100]}")
print()

# ── 글자 단위 토크나이저 ──
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"어휘 크기: {vocab_size}개 (글자 단위)")
print(f"어휘: {''.join(chars[:30])}...")
print()

# ── 학습/검증 데이터 분리 ──
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"학습 데이터: {len(train_data):,} 토큰")
print(f"검증 데이터: {len(val_data):,} 토큰")
print()

# ── 배치 생성 ──
def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - max_len, (batch_size,))
    x = torch.stack([data_split[i:i+max_len] for i in ix]).to(device)
    y = torch.stack([data_split[i+1:i+max_len+1] for i in ix]).to(device)
    return x, y

# ── 모델 생성 ──
model = GPT(vocab_size, d_model, n_heads, n_layers, max_len).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"모델 파라미터: {n_params:,}개")

# initial loss 확인: ln(vocab_size) 근처여야 함
x, y = get_batch("train")
_, loss = model(x, y)
import math
print(f"초기 loss: {loss.item():.4f}  (이론값 ln({vocab_size})={math.log(vocab_size):.4f})")
print()

# ── 학습 ──
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("학습 시작!")
print("-" * 50)
for step in range(max_steps):
    # forward
    x, y = get_batch("train")
    logits, loss = model(x, y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0 or step == max_steps - 1:
        # 검증 loss
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch("val")
            _, val_loss = model(xv, yv)
        model.train()

        print(f"step {step:5d}  train_loss={loss.item():.4f}  val_loss={val_loss.item():.4f}")

        # 중간 생성 샘플
        if step >= 1000:
            ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample = model.generate(ctx, max_new_tokens=100)
            print(f"  생성: {decode(sample[0].tolist())[:80]}...")
            print()

print("-" * 50)
print("학습 완료!\n")

# ── 최종 생성 ──
print("=" * 50)
print("최종 생성 결과:")
print("=" * 50)
model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
sample = model.generate(ctx, max_new_tokens=300, temperature=0.8)
print(decode(sample[0].tolist()))

# ── 체크포인트 저장 ──
torch.save(model.state_dict(), "nanogpt_shakespeare.pt")
print("\n모델 저장 완료: nanogpt_shakespeare.pt")
