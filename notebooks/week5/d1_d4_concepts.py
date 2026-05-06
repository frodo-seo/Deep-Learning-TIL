"""Week 5 · D1~D4 — 최적화, 초기화, 정규화 개념 실습"""

import torch
import torch.nn as nn

# ══════════════════════════════════════
# D1: SGD vs Momentum vs Adam
# ══════════════════════════════════════
print("=== D1: 옵티마이저 비교 ===\n")

# 울퉁불퉁한 함수: f(x) = x^4 - 3x^2 + x (local minimum 있음)
def train_1d(optim_name, lr=0.01, steps=100):
    x = torch.tensor(3.0, requires_grad=True)
    if optim_name == "SGD":
        opt = torch.optim.SGD([x], lr=lr)
    elif optim_name == "Momentum":
        opt = torch.optim.SGD([x], lr=lr, momentum=0.9)
    elif optim_name == "Adam":
        opt = torch.optim.Adam([x], lr=lr)

    for step in range(steps):
        loss = x**4 - 3*x**2 + x
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"  {optim_name:10s} → x={x.item():.4f}, f(x)={loss.item():.4f}")

train_1d("SGD")
train_1d("Momentum")
train_1d("Adam", lr=0.1)
print()

# ══════════════════════════════════════
# D2: 가중치 초기화가 왜 중요한가
# ══════════════════════════════════════
print("=== D2: 가중치 초기화 ===\n")

torch.manual_seed(42)

# 나쁜 초기화: 너무 큰 값
bad = nn.Linear(100, 100)
nn.init.normal_(bad.weight, std=10.0)
x = torch.randn(1, 100)
out = bad(x)
print(f"  나쁜 초기화 (std=10): 출력 평균={out.mean().item():.1f}, 표준편차={out.std().item():.1f}")
# → 값이 폭발해서 학습 불안정

# Xavier 초기화: 입출력 크기에 맞게 조절
good = nn.Linear(100, 100)
nn.init.xavier_normal_(good.weight)
out = good(x)
print(f"  Xavier 초기화:       출력 평균={out.mean().item():.2f}, 표준편차={out.std().item():.2f}")
# → 분산이 유지되어 안정적
print()

# ══════════════════════════════════════
# D3: LayerNorm vs BatchNorm
# ══════════════════════════════════════
print("=== D3: LayerNorm vs BatchNorm ===\n")

# BatchNorm: 배치 방향으로 정규화 (배치 내 같은 뉴런끼리)
# LayerNorm: 레이어 방향으로 정규화 (한 샘플 내 모든 뉴런끼리)

x = torch.randn(4, 8)  # 배치 4, 특성 8

bn = nn.BatchNorm1d(8)
ln = nn.LayerNorm(8)

bn_out = bn(x)
ln_out = ln(x)

print(f"  BatchNorm: 열(뉴런)별 평균≈0 →  {bn_out[:, 0].mean().item():.4f}")
print(f"  LayerNorm: 행(샘플)별 평균≈0 →  {ln_out[0, :].mean().item():.4f}")
print()
print("  Transformer는 LayerNorm을 씀")
print("  이유: 시퀀스 길이가 가변이라 배치 통계가 불안정")
print("        + 추론 시 배치 크기 1이어도 정상 작동")
print()

# ══════════════════════════════════════
# D4: Dropout
# ══════════════════════════════════════
print("=== D4: Dropout ===\n")

dropout = nn.Dropout(p=0.3)  # 30% 확률로 뉴런 꺼짐

x = torch.ones(1, 10)

# 학습 모드
dropout.train()
print(f"  학습 시: {dropout(x)}")  # 일부가 0으로

# 추론 모드
dropout.eval()
print(f"  추론 시: {dropout(x)}")  # 전부 살아있음

print()
print("  왜? 특정 뉴런에 의존하지 않게 → 과적합 방지")
print("  추론 때는 모든 뉴런 사용 (단, 학습 때 꺼진 비율만큼 스케일 조정)")
