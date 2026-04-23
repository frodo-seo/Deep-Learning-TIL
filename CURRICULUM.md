# Deep Learning 커리큘럼 — Transformer 이해 + 직접 학습 (10주 기초 강화판)

> **최종 목표**
> 1. Transformer의 **이론을 수식·구조까지 탄탄하게** 이해한다.
> 2. 작은 Transformer(nanoGPT 스타일)를 **직접 학습**시켜 Tiny Shakespeare에서 loss가 내려가고 그럴듯한 텍스트가 생성되는 것을 본인 손으로 확인한다.
>
> **학습 시간**: 하루 30분 (주 5일) · 총 **10주**
> **스택**: Python, PyTorch
> **1순위 레퍼런스**: Andrej Karpathy — `nn-zero-to-hero`, `micrograd`, `makemore`, `nanoGPT`, "Let's build GPT" 영상

---

## 설계 원칙

- **기초 탄탄**: 수학(선형대수·미적분)·역전파·MLP·최적화까지 **5주를 온전히 투입**. 이 위에서만 Attention이 "수식으로" 읽힌다.
- **수식은 손으로**: softmax+CE, 역전파, scaled dot-product attention — 최소 한 번은 노트에 손으로.
- **돌려봐야 안다**: 매 구간마다 돌아가는 코드 1개. 마지막 2주는 순수 학습·디버깅.
- **Karpathy 먼저**: 영상 1개가 교과서 한 챕터를 대체할 때가 많음. 있으면 1순위.
- **막히면 남기기**: 이해 안 된 수식/코드는 `til/YYYY-MM-DD.md`에 "오늘 막힌 것"으로. 다음날 같이 풀자.

---

## 전체 로드맵

```
[수학·역전파·MLP 기초]                  [언어모델 → Attention → Transformer]       [학습 🎯]
 Week 1–5 (5주)                          Week 6–8 (3주)                             Week 9–10 (2주)
```

---

## Week 1 — 선형대수 직관 + NumPy 감각

> Transformer 수식이 "차원으로 읽히는" 수준이 목표. NumPy도 같이 친해진다.

- **D1** 벡터·행렬·내적의 기하적 의미 (3Blue1Brown *Essence of Linear Algebra* Ep.1–2) + NumPy `np.array`, shape, `.T`
- **D2** 행렬곱 = 선형변환의 합성 (Ep.3–4) + NumPy `@` / `np.matmul` 실습
- **D3** NumPy 브로드캐스팅·슬라이싱·`reshape` 집중 — 헷갈리는 지점 손으로 그려보기
- **D4** 전치·역행렬·랭크 직관 (SVD는 "개념만") — NumPy로 짧게 확인
- **D5** 텐서 차원 `(B, T, C)` 읽기 연습 + Attention 수식 `softmax(QKᵀ/√d_k)V`를 **차원만으로** 분해

> *einsum은 Week 3에서 PyTorch와 함께 다룸. Week 1은 기본 `@` / `*` / 브로드캐스팅에 집중.*

---

## Week 2 — 미적분 + Softmax/Cross Entropy

> 역전파의 뼈대(연쇄법칙)와 분류의 손실함수를 **손으로** 유도.

- **D1** 편미분, 그라디언트 = "가장 가파르게 오르는 방향" (3B1B *Essence of Calculus* 일부)
- **D2** 연쇄법칙 — 2-layer MLP 예시 손유도
- **D3** Softmax, Log-Softmax, 수치안정성 (왜 max를 빼는가)
- **D4** Cross Entropy 유도, Softmax+CE의 깔끔한 그라디언트 `ŷ - y`
- **D5** KL divergence 직관, MLE = CE 최소화 관점

---

## Week 3 — PyTorch & autograd

> Python은 익숙하다는 전제. PyTorch API와 **자동미분**만 빠르게.

- **D1** `torch.Tensor` 기본, NumPy와 차이, `dtype`/`device`
- **D2** autograd: `requires_grad`, `.backward()`, `.grad` — Week 2 손유도와 수치 일치 확인
- **D3** `nn.Module`, `nn.Linear`로 선형회귀 바닥부터
- **D4** `optim.SGD`/`Adam`, 표준 학습 루프 템플릿 암기
- **D5** GPU `.to(device)`, 재현성(seed), 체크포인트 저장/로드

---

## Week 4 — micrograd & 역전파 직접 만들기

> "자동으로 되는 것"이 아니라 **손으로 그려지는 것**으로.

- **D1** Karpathy *micrograd* 영상 전반 — 스칼라 자동미분 원리
- **D2** 역전파 수식 손유도 (2-layer MLP 끝까지, Week 2와 연결)
- **D3** micrograd 엔진 따라 만들기 — `add` / `mul` / `tanh`의 backward
- **D4** 만든 엔진으로 작은 분류 문제 학습 → loss 내려가는 것 확인
- **D5** PyTorch autograd와 결과 비교 (그라디언트 값 일치하는가)

📘 자료: `karpathy/micrograd` + "The spelled-out intro to neural networks and backpropagation"

---

## Week 5 — MLP, 최적화, 정규화

> "왜 학습이 되는가"의 디테일. LN이 왜 Transformer 표준인지도 이 주에.

- **D1** SGD / Momentum / Adam 차이 — 그림으로 비교
- **D2** 가중치 초기화: Xavier / He — 왜 필요한가 (분산 유지 관점)
- **D3** **LayerNorm vs BatchNorm** — 왜 Transformer는 LN인가 (시퀀스 길이·배치 독립성)
- **D4** Dropout, Weight decay 직관
- **D5** PyTorch로 2-layer MLP MNIST 학습, loss curve 해석

---

## Week 6 — makemore: 언어모델 감 잡기

> "다음 토큰 확률 분포" 예측 구조를 체화.

- **D1** Karpathy *makemore* ep.1 — bigram 언어모델, 카운트 기반 → 신경망 기반
- **D2** `nn.Embedding` 실제 동작 (lookup table), 임베딩 차원 직관
- **D3** makemore MLP 버전 — 문맥 창 + 임베딩 결합
- **D4** 학습률 감, LR finder, 과적합/미적합 신호
- **D5** 생성해보기, "logits → softmax → 샘플링" 파이프라인 체화

📘 자료: `karpathy/makemore` + "Building makemore" 시리즈

---

## Week 7 — Attention 수식 완전정복

> **이 커리큘럼의 심장.** 천천히, 수식 중심으로.

- **D1** Query/Key/Value 개념, 내적 = 유사도
- **D2** Scaled Dot-Product `softmax(QKᵀ/√d_k)V` — **왜 √d_k로 나누는가** (분산 분석 손유도)
- **D3** Masking: causal mask(미래 가리기), padding mask
- **D4** Multi-Head — 왜 쪼개서 여러 번? 표현 공간 분할 직관
- **D5** 복잡도 O(n²·d) 분석, 메모리 병목 이해

---

## Week 8 — Transformer 블록 + 논문 정독

> Attention 옆에 붙는 조각들 + *Attention Is All You Need* 직접.

- **D1** Positional Encoding (sinusoidal 수식 감 — RoPE/learned는 이름만)
- **D2** FFN, Residual, LayerNorm의 역할
- **D3** Pre-LN vs Post-LN — 학습 안정성 차이
- **D4** 논문 *"Attention Is All You Need"* — Model Architecture 섹션 수식 한 줄씩
- **D5** Decoder-only(GPT 계열) 구조 총정리 → `til/transformer_arch.md`

---

## Week 9 — nanoGPT 코드 해부

> 읽을 수 있다 → **만들 수 있다**로 건너가는 주.

- **D1** Karpathy *"Let's build GPT"* 영상 전반부 — 데이터, tokenizer
- **D2** `nanoGPT/model.py` — `CausalSelfAttention` 줄줄이 읽기
- **D3** `Block`(Attention + FFN + residual + LN) 조립
- **D4** `train.py` — 학습 루프, AdamW, weight decay 적용 방식
- **D5** `generate()` — greedy / top-k / temperature 구현 읽기

📘 자료: `karpathy/nanoGPT` + "Let's build GPT: from scratch, in code, spelled out"

---

## Week 10 — Tiny Shakespeare 학습 🎯

> **이 주가 이 커리큘럼의 끝점.** loss가 내려가는 걸 네 눈으로 본다.

- **D1** 환경 세팅 — 데이터 다운, tokenizer, 첫 forward pass
- **D2** 학습 시작 — 첫 1000 step, initial loss가 `ln(vocab_size)` 근처인지 확인
- **D3** Warmup + cosine schedule + gradient clipping + mixed precision(`amp`) 붙이기, loss curve 해석
- **D4** 학습 완료 → 텍스트 생성. 디코딩 실험 (greedy / top-k / top-p / temperature)
- **D5** 한 페이지 요약 **"Transformer: 입력부터 loss까지"** + 회고

→ 산출물: `projects/nanogpt/` 에 학습 스크립트·loss 로그·샘플 생성 결과

---

## 매일 30분 루틴

- **5분** — 어제 `til/` 읽기
- **20분** — 오늘 주제 (영상 1개 / 논문 문단 / 코드 중 택1)
- **5분** — `til/YYYY-MM-DD.md`에 3줄 요약 + 질문 1개

---

## 레포 구조

```
Deep-Learning-TIL/
├── CURRICULUM.md        # 이 문서
├── til/                 # 매일 기록
├── notebooks/           # 주차별 실습 (Week별 폴더)
├── papers/              # 논문 요약 (Week 8: Attention Is All You Need)
└── projects/
    └── nanogpt/         # Week 9–10 산출물
```

---

## 필수 레퍼런스

- **Karpathy GitHub** (1순위)
  - `karpathy/nn-zero-to-hero` — 전체 영상 시리즈
  - `karpathy/micrograd` — Week 4
  - `karpathy/makemore` — Week 6
  - `karpathy/nanoGPT` — Week 9–10
- **영상**: "The spelled-out intro to backprop" / "Building makemore" / "Let's build GPT"
- **논문**: Vaswani et al., *Attention Is All You Need* (2017)
- **수학 보조**: 3Blue1Brown *Essence of Linear Algebra* / *Essence of Calculus*

---

## 막힐 때 규칙

논문 수식이나 코드에서 막히면 넘어가지 말고 `til/`에 **"오늘 막힌 것"** 으로 남겨두기.
다음날 같이 풀자 — 이 레포에서 내가 옆에 있을게.

---

## 다음 목표 후보 (10주 완주 후)

이 10주를 끝내면 다음 중에서 고르자:
- **VLM 트랙**: CLIP → LLaVA 계열 (원래 23주 플랜의 Phase 6–9, git history에 있음).
- **스케일업**: 더 큰 모델·데이터로 nanoGPT 확장, FlashAttention·KV cache 등 심화.
- **논문 재현**: GPT-2 재현, 최신 Transformer 변형 논문 하나.
