"""Week 3 · D1 — torch.Tensor 기본"""

import numpy as np
import torch

# ── 1. 텐서 생성 ──
a = torch.tensor([1.0, 2.0, 3.0])
print("텐서:", a)
print("shape:", a.shape)
print("dtype:", a.dtype)

# ── 2. NumPy와 비교 ──
a_np = np.array([1.0, 2.0, 3.0])
a_pt = torch.from_numpy(a_np)       # numpy → tensor
back = a_pt.numpy()                  # tensor → numpy
print("\nnumpy → tensor:", a_pt)
print("tensor → numpy:", back)

# ── 3. 기본 연산 (NumPy랑 거의 동일) ──
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print("\n덧셈:", x + y)
print("곱셈:", x * y)           # element-wise
print("내적:", x @ y)            # dot product (Week 1!)
print("행렬곱:", x.reshape(3,1) @ y.reshape(1,3))  # outer product

# ── 4. shape 다루기 ──
m = torch.randn(2, 3, 4)   # (B, T, C) 느낌
print("\n3D 텐서 shape:", m.shape)
print("reshape:", m.reshape(6, 4).shape)
print("transpose:", m.transpose(1, 2).shape)  # (2, 4, 3)

# ── 5. 핵심 차이: requires_grad ──
w = torch.tensor([2.0], requires_grad=True)
print("\nrequires_grad:", w.requires_grad)
print("→ 이게 있으면 PyTorch가 자동으로 역전파 해줌 (D2에서 실습)")

# ── 6. device ──
print("\ndevice:", a.device)   # cpu
# GPU가 있으면: a.to('cuda')
