"""Week 3 · D3 — nn.Module + nn.Linear로 선형회귀 바닥부터"""

import torch
import torch.nn as nn

# ── 1. 데이터: y = 3x + 1 (이걸 모델이 알아내야 함) ──
torch.manual_seed(42)
x = torch.linspace(0, 10, 50).reshape(-1, 1)   # (50, 1)
y = 3 * x + 1 + torch.randn(50, 1) * 0.5       # 약간의 노이즈

# ── 2. 모델 정의 ──
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)   # 입력 1개 → 출력 1개 (w, b 자동 생성)

    def forward(self, x):
        return self.linear(x)           # z = wx + b

model = LinearModel()
print("초기 w:", model.linear.weight.item())
print("초기 b:", model.linear.bias.item())
print()

# ── 3. 손실함수 + 옵티마이저 ──
criterion = nn.MSELoss()                # (예측 - 정답)^2 의 평균
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ── 4. 학습 루프 ──
for epoch in range(100):
    # forward
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # backward
    optimizer.zero_grad()   # 이전 기울기 초기화 (안 하면 누적됨!)
    loss.backward()         # 역전파 → w.grad, b.grad 계산
    optimizer.step()        # w = w - lr * grad (경사하강법)

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}")

# ── 5. 결과: 3과 1을 찾았는가? ──
print()
print(f"학습된 w: {model.linear.weight.item():.4f}  (정답: 3)")
print(f"학습된 b: {model.linear.bias.item():.4f}  (정답: 1)")
