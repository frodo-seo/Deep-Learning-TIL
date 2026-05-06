"""Week 3 · D4 — SGD vs Adam 비교"""

import torch
import torch.nn as nn

torch.manual_seed(42)
x = torch.linspace(0, 10, 50).reshape(-1, 1)
y = 3 * x + 1 + torch.randn(50, 1) * 0.5

def train(optimizer_name, lr, epochs=100):
    torch.manual_seed(42)
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}")

    w = model.weight.item()
    b = model.bias.item()
    print(f"  결과: w={w:.4f}  b={b:.4f}\n")
    return losses

# ── SGD (lr=0.01) ──
print("=== SGD (lr=0.01) ===")
sgd_losses = train("SGD", lr=0.01)

# ── Adam (lr=0.01) ──
print("=== Adam (lr=0.01) ===")
adam_losses = train("Adam", lr=0.01)

# ── 비교 ──
print("=== 최종 loss 비교 ===")
print(f"SGD  최종 loss: {sgd_losses[-1]:.4f}")
print(f"Adam 최종 loss: {adam_losses[-1]:.4f}")
print()
print("Adam이 보통 더 빠르게 수렴합니다.")
print("이유: Adam은 기울기의 '속도'와 '가속도'를 함께 고려해서")
print("      적응적으로 학습률을 조절하기 때문.")
