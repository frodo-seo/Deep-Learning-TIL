# Deep-Learning-TIL

Transformer를 수식부터 구현까지 이해하기 위한 딥러닝 학습 기록. 10주 분량 커리큘럼을 2세션에 압축 완주.

## 커리큘럼

| Week | 주제 | 핵심 |
|------|------|------|
| 1 | 선형대수 | 벡터, 행렬곱, 내적, SVD, `(B,T,C)` 차원 |
| 2 | 미적분 | 편미분, 연쇄법칙, Softmax+CE, MLE |
| 3 | PyTorch & autograd | Tensor, backward(), 학습 루프 4줄 |
| 4 | micrograd | 자동미분 엔진 직접 구현 |
| 5 | 최적화, 정규화 | Adam, Xavier 초기화, LayerNorm, Dropout |
| 6 | makemore | Bigram, Embedding, MLP 언어모델 |
| 7 | Attention | Q,K,V, Scaled Dot-Product, Multi-Head, Causal Mask |
| 8 | Transformer 블록 | Positional Encoding, FFN, Residual, Pre-LN |
| 9-10 | nanoGPT | 구현 + Tiny Shakespeare 학습 → 텍스트 생성 |

## 레포 구조

```
Deep-Learning-TIL/
├── CURRICULUM.md              # 10주 커리큘럼 상세
├── til/                       # 매일 학습 기록
│   ├── 2026-04-25.md          # Week 1 ~ Week 2 D2
│   ├── 2026-05-06.md          # Week 2 D3 ~ Week 10 완주
│   └── NEXT.md                # 다음 세션 시작점
├── notebooks/                 # 주차별 실습 코드
│   ├── week3/                 # PyTorch 기초
│   ├── week4/                 # micrograd 엔진
│   ├── week5/                 # 최적화/정규화
│   ├── week6/                 # bigram + MLP 언어모델
│   └── week8/                 # GPT 구조 시각화
└── projects/
    └── nanogpt/               # nanoGPT 구현 + 학습
        ├── model.py           # GPT 모델 (Attention, Block, GPT)
        └── train.py           # Shakespeare 학습 스크립트
```

## 스택

- Python 3.14
- PyTorch 2.11
- NumPy, Matplotlib, Jupyter

## 레퍼런스

- [Karpathy nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [3Blue1Brown Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
