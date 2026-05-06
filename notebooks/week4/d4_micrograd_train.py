"""Week 4 · D4 — micrograd 엔진으로 실제 학습"""

# d1에서 만든 Value 클래스 그대로 가져오기
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from d1_micrograd_engine import Value

import random
random.seed(42)

# ── 데이터: y = 3x + 1 을 학습으로 찾기 ──
data = [(1, 4), (2, 7), (3, 10), (4, 13), (5, 16)]
# x=1→y=4, x=2→y=7 ... (3*x+1)

# ── 랜덤 초기값 ──
w = Value(random.uniform(-1, 1))   # 정답: 3
b = Value(random.uniform(-1, 1))   # 정답: 1
lr = 0.01

print(f"초기값: w={w.data:.4f}, b={b.data:.4f}")
print(f"목표:   w=3.0000, b=1.0000\n")

# ── 학습 루프 ──
for epoch in range(200):
    # 1. 전체 데이터에 대해 loss 계산
    total_loss = Value(0.0)
    for x_val, y_val in data:
        x = Value(x_val)
        y_true = Value(y_val)
        y_pred = w * x + b              # forward
        total_loss = total_loss + (y_pred - y_true) ** 2

    # 2. 역전파
    # 기울기 초기화 (PyTorch의 zero_grad와 같은 역할!)
    w.grad = 0.0
    b.grad = 0.0
    total_loss.backward()

    # 3. 파라미터 업데이트 (경사하강법)
    w.data -= lr * w.grad
    b.data -= lr * b.grad

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss={total_loss.data:.4f}  w={w.data:.4f}  b={b.data:.4f}")

print(f"\n최종: w={w.data:.4f}  b={b.data:.4f}")
print(f"정답: w=3.0000  b=1.0000")
