"""Week 3 · D2 — autograd: 자동 미분"""

import torch

# ── 1. 간단한 예: y = x^2 의 미분 ──
x = torch.tensor(3.0, requires_grad=True)   # "이 텐서의 기울기를 추적해줘"
y = x ** 2                                    # y = 9

y.backward()                                  # 역전파 실행!
print("x =", x.item())
print("y = x^2 =", y.item())
print("dy/dx =", x.grad.item())               # 손으로 구하면 2x = 6
print()

# ── 2. Week 2 D3 복습: 연쇄법칙 자동 검증 ──
# y = (3x + 2)^2   →   손유도: dy/dx = 6(3x+2)
x = torch.tensor(1.0, requires_grad=True)
y = (3*x + 2) ** 2

y.backward()
print("y = (3x+2)^2,  x=1")
print("PyTorch dy/dx =", x.grad.item())       # 자동
print("손유도 dy/dx   =", 6*(3*1+2))            # 6*(3+2) = 30
print()

# ── 3. 신경망 스타일: z = wx + b ──
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)                         # 입력 (기울기 불필요)
y_true = torch.tensor(10.0)                   # 정답

# forward
z = w * x + b          # 2*3 + 1 = 7
loss = (z - y_true)**2  # (7 - 10)^2 = 9

# backward
loss.backward()

print("z = wx + b =", z.item())
print("loss = (z - y_true)^2 =", loss.item())
print()
print("dL/dw =", w.grad.item())   # 손유도: 2(z-y_true)*x = 2*(-3)*3 = -18
print("dL/db =", b.grad.item())   # 손유도: 2(z-y_true)*1 = 2*(-3) = -6
print()
print("검증 (손유도):")
print("dL/dw = 2*(z-y)*x = 2*(-3)*3 =", 2*(-3)*3)
print("dL/db = 2*(z-y)*1 = 2*(-3)*1 =", 2*(-3)*1)
