"""Week 3 · D5 — GPU, 재현성, 체크포인트"""

import torch
import torch.nn as nn

# ── 1. Device: CPU vs GPU ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 device: {device}")

# 텐서를 device로 보내기
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(f"텐서 위치: {x.device}")
# 모델도 마찬가지: model.to(device)
# → 모델과 데이터가 같은 device에 있어야 연산 가능!
print()

# ── 2. 재현성 (Seed) ──
# 같은 seed → 같은 랜덤 값 → 같은 결과
torch.manual_seed(42)
a = torch.randn(3)
print("seed 42 첫 번째:", a)

torch.manual_seed(42)
b = torch.randn(3)
print("seed 42 두 번째:", b)
print("동일한가?", torch.equal(a, b))   # True!
print()

# 왜 중요? → 실험 결과를 재현할 수 있어야 디버깅 가능
# "어제 loss가 낮았는데 오늘 다시 돌리니 다르다" → seed 고정으로 해결

# ── 3. 체크포인트 저장/로드 ──
# 학습 중간에 모델 저장 → 나중에 이어서 학습 or 배포
model = nn.Linear(1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 저장
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 50,
}, "checkpoint.pt")
print("체크포인트 저장 완료")

# 로드
checkpoint = torch.load("checkpoint.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
print(f"체크포인트 로드 완료 (epoch {epoch}부터 재개 가능)")
print()

# ── 정리 ──
print("=== Week 3 핵심 정리 ===")
print("D1: torch.Tensor — NumPy + autograd + GPU")
print("D2: autograd — requires_grad + backward() = 자동 역전파")
print("D3: nn.Module — 모델 구조 정의, nn.Linear = wx+b")
print("D4: optimizer — SGD(단순), Adam(적응적), 학습 루프 4줄")
print("D5: device(GPU), seed(재현성), checkpoint(저장/로드)")
