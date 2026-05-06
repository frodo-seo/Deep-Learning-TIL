"""Week 4 · D1~D3 — micrograd 엔진 직접 만들기"""

class Value:
    """스칼라 값 하나 + 자동미분 기능"""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0                # 이 값의 기울기 (역전파로 채워짐)
        self._backward = lambda: None  # 역전파 함수 (연산마다 다름)
        self._children = set(_children)
        self._op = _op                 # 디버깅용: 어떤 연산으로 만들어졌나

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # 덧셈의 미분: 둘 다 1을 곱해서 전달
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 곱셈의 미분: 상대방 값을 곱해서 전달
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            self.grad += n * (self.data ** (n-1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        """역전파: 출력에서 입력 방향으로 기울기 전파"""
        # 위상 정렬: 의존성 순서대로 정렬
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0  # 자기 자신에 대한 미분 = 1
        for v in reversed(topo):
            v._backward()


# ══════════════════════════════════════
# 테스트 1: 간단한 수식
# ══════════════════════════════════════
print("=== 테스트 1: y = (2x + 3)^2,  x=1 ===")
x = Value(1.0)
y = (2*x + 3) ** 2    # (2+3)^2 = 25

y.backward()
print(f"y = {y.data}")
print(f"dy/dx = {x.grad}")
# 손유도: dy/dx = 2*2*(2x+3) = 4*(2+3) = 20
print(f"손유도: 4*(2*1+3) = {4*(2*1+3)}")
print()

# ══════════════════════════════════════
# 테스트 2: 신경망 스타일 (Week 2 복습)
# ══════════════════════════════════════
print("=== 테스트 2: 신경망 z=wx+b, loss=(z-y)^2 ===")
w = Value(2.0)
b = Value(1.0)
x_in = Value(3.0)
y_true = Value(10.0)

z = w * x_in + b         # 2*3 + 1 = 7
loss = (z - y_true) ** 2  # (7-10)^2 = 9

loss.backward()
print(f"z = {z.data},  loss = {loss.data}")
print(f"dL/dw = {w.grad}  (손유도: {2*(7-10)*3})")
print(f"dL/db = {b.grad}  (손유도: {2*(7-10)*1})")
print()

# ══════════════════════════════════════
# 테스트 3: PyTorch와 결과 비교
# ══════════════════════════════════════
import torch

print("=== PyTorch 검증 ===")
w_pt = torch.tensor(2.0, requires_grad=True)
b_pt = torch.tensor(1.0, requires_grad=True)
x_pt = torch.tensor(3.0)
y_pt = torch.tensor(10.0)

z_pt = w_pt * x_pt + b_pt
loss_pt = (z_pt - y_pt) ** 2
loss_pt.backward()

print(f"PyTorch dL/dw = {w_pt.grad.item()}")
print(f"우리    dL/dw = {w.grad}")
print(f"일치? {w_pt.grad.item() == w.grad}")
print()
print(f"PyTorch dL/db = {b_pt.grad.item()}")
print(f"우리    dL/db = {b.grad}")
print(f"일치? {b_pt.grad.item() == b.grad}")
