# Deep Learning 커리큘럼 — Transformer & VLM 마스터 루트

> **최종 목표**
> 1. Transformer를 수식·구현·학습 동역학까지 **완벽히** 이해한다.
> 2. 그 위에서 Vision-Language Model(VLM)의 구조와 학습 방식을 이해하고 직접 작게 구현해본다.
>
> **학습 시간**: 하루 30분 (주 5일) · 총 **약 23주**
> **스택**: Python, PyTorch, HuggingFace `transformers`, `datasets`

---

## 전체 로드맵

```
[기초 다지기]        → [Transformer 완전정복]   → [비전 파트]        → [VLM]
 Phase 1–3 (9주)      Phase 4–5 (6주)           Phase 6 (3주)       Phase 7–9 (7주)
```

모든 Phase는 "이론 → 손으로 유도 → PyTorch 구현 → 논문 확인" 4단을 반복.

---

## Phase 1. 수학 복습 (2주 / Week 1–2)

> 사용자 상태: Python 익숙 / PyTorch 생소 / 선형대수·미적분 복습 필요.
> → Python 기초는 건너뛰고, 수학은 "Transformer 수식이 막힘없이 읽힐 정도"까지만 복습.

- **Week 1 — 선형대수 복습**
  - D1 벡터/행렬/내적의 기하적 의미 (3Blue1Brown Ep.1–3)
  - D2 행렬곱 = 선형변환, 합성 (Ep.4)
  - D3 텐서 차원 감각: `(batch, seq, dim)` 읽는 법, NumPy `einsum` 5문제
  - D4 전치·역행렬·랭크 직관 / SVD는 "개념만"
  - D5 주간 정리: Attention 수식 `softmax(QKᵀ/√d)V`를 차원으로만 분해해보기
- **Week 2 — 미적분·확률 복습 + Softmax/CE**
  - D1 편미분, 그라디언트 = "가장 가파르게 오르는 방향"
  - D2 연쇄법칙 — 역전파의 뼈대 (손으로 2-layer 예시 1개)
  - D3 Softmax, Log-Softmax, 수치안정성 (왜 max를 빼는가)
  - D4 Cross Entropy 유도, Softmax+CE의 깔끔한 그라디언트
  - D5 KL divergence 직관, MLE = CE 최소화 관점

📘 추천: 3Blue1Brown "Essence of Linear Algebra" + "Essence of Calculus" (각 10분 내외).
   매일 영상 1개 + 노트 3줄이면 30분 안에 끝남.

---

## Phase 1.5. PyTorch 집중 입문 (1주 / Week 3)

> Python은 익숙하니 **PyTorch API와 autograd만** 빠르게.

- D1 `torch.Tensor` 기본, NumPy와의 차이, `dtype`/`device`
- D2 autograd: `requires_grad`, `.backward()`, `.grad` — 수식 손유도 결과와 일치 확인
- D3 `nn.Module` 작성법, `nn.Linear`로 선형회귀를 바닥부터
- D4 `optim.SGD`/`Adam`, 표준 학습 루프 템플릿 암기
- D5 GPU(`.to(device)`), 재현성(seed), 체크포인트 저장/로드
- (주말 보너스) `DataLoader`와 `Dataset` 감 잡기

---

## Phase 2. 신경망 핵심 (3주 / Week 4–6)

"왜 이렇게 동작하는지"를 설명할 수 있는 수준이 목표.

- **Week 4 — MLP & 역전파**
  - 순전파/역전파 수식 직접 유도 (한 번은 손으로)
  - NumPy로 2-layer MLP 구현 → MNIST 학습
- **Week 5 — 최적화 & 정규화**
  - SGD/Momentum/Adam 차이
  - 가중치 초기화(Xavier/He), LayerNorm vs BatchNorm
  - Dropout, Weight decay
- **Week 6 — 임베딩 & 언어모델 감 잡기**
  - 토큰/임베딩 개념, `nn.Embedding`
  - n-gram 언어모델 → 신경 언어모델 개요
  - 문자 단위 LM을 작은 MLP로 (Karpathy "makemore" 1편)

---

## Phase 3. 시퀀스 모델 맛보기 (1주 / Week 7)

Transformer의 **이전 세대**를 빠르게 훑어 대비 포인트를 남김.

- D1 RNN의 아이디어와 한계(기울기 소실)
- D2 LSTM/GRU 게이트 직관
- D3 Seq2Seq의 병목 문제 → Attention 등장 배경
- D4 Bahdanau Attention 수식
- D5 정리: "왜 Transformer가 필요했는가?" 한 쪽 요약

---

## Phase 4. Transformer 완전정복 — 이론 (3주 / Week 8–10)

이 구간이 **이 커리큘럼의 심장**. 천천히, 확실히.

- **Week 8 — Attention의 모든 것**
  - D1 Query/Key/Value 개념, 내적 유사도
  - D2 Scaled Dot-Product Attention 수식 유도 (왜 √d_k로 나누나)
  - D3 Multi-Head Attention: 왜 나눠서 여러 번?
  - D4 Masking: padding mask, causal mask
  - D5 복잡도 O(n²d) 분석, 메모리 병목 이해
- **Week 9 — Transformer 블록**
  - D1 Positional Encoding (sinusoidal vs learned vs RoPE 예고)
  - D2 Feed-Forward Network, residual + LayerNorm
  - D3 Pre-LN vs Post-LN 차이와 학습 안정성
  - D4 Encoder 블록 완전 분해
  - D5 Decoder 블록 + cross-attention
- **Week 10 — 논문 정독: "Attention Is All You Need"**
  - D1 Abstract + Introduction + Model Architecture 섹션
  - D2 Attention 섹션 수식 한 줄씩 체크
  - D3 Training 섹션 (optimizer, warmup, label smoothing)
  - D4 Results/Ablation 표 읽기
  - D5 한 페이지 요약본 작성 → `til/transformer_paper.md`

---

## Phase 5. Transformer 완전정복 — 구현 & 변형 (3주 / Week 11–13)

"읽을 수 있다"에서 "만들 수 있다"로.

- **Week 11 — 밑바닥부터 구현**
  - D1 Tokenizer: BPE 개념과 HuggingFace `tokenizers` 사용
  - D2 `MultiHeadAttention` 모듈 직접 구현
  - D3 `TransformerBlock` (MHA + FFN + residual + LN)
  - D4 전체 Decoder-only 모델 조립 (nanoGPT 스타일)
  - D5 Tiny Shakespeare로 학습 돌려보기
- **Week 12 — 학습 동역학 이해**
  - D1 Warmup + cosine schedule, 왜 필요한가
  - D2 Gradient clipping, mixed precision(`amp`)
  - D3 Loss curve 읽는 법, overfitting 징후
  - D4 생성 디코딩: greedy / top-k / top-p / temperature
  - D5 내 모델로 텍스트 생성해보기
- **Week 13 — Encoder/Decoder 계열 비교**
  - D1 GPT (Decoder-only, causal LM)
  - D2 BERT (Encoder-only, Masked LM)
  - D3 T5 (Encoder-Decoder, span corruption)
  - D4 HuggingFace로 BERT 파인튜닝 체험 (감정분석)
  - D5 세 계열을 한 표로 비교 정리

---

## Phase 6. 비전 파트 — CNN은 얇게, ViT는 두껍게 (3주 / Week 14–16)

VLM으로 가려면 이미지를 "토큰처럼" 다룰 줄 알아야 함.

- **Week 14 — CNN 속성 이해 (얇게)**
  - D1 Convolution/pooling 직관, 수용영역
  - D2 ResNet의 skip connection이 왜 중요했나
  - D3 torchvision pretrained로 특징 추출 체험
  - D4 이미지 전처리·정규화·augmentation 파이프라인
  - D5 CNN vs Transformer 귀납적 편향 비교
- **Week 15 — Vision Transformer (ViT)**
  - D1 이미지 → 패치 → 토큰 시퀀스 변환
  - D2 CLS 토큰, Positional Embedding in ViT
  - D3 ViT 논문 ("An Image is Worth 16x16 Words") 정독
  - D4 `timm`으로 pretrained ViT 불러와 추론
  - D5 패치 임베딩 모듈 직접 구현
- **Week 16 — 현대 비전 백본 개요**
  - D1 Swin Transformer 아이디어 (윈도우 어텐션)
  - D2 MAE (Masked Autoencoder) 사전학습
  - D3 DINO/DINOv2 자기지도 학습 개요
  - D4 비전 백본 선택 가이드 정리
  - D5 Phase 6 복습 + 손그림 정리

---

## Phase 7. VLM의 기초 — CLIP과 대조학습 (2주 / Week 17–18)

VLM의 "공통 임베딩 공간" 개념이 이 구간에서 자리잡음.

- **Week 17 — CLIP 완전 이해**
  - D1 Contrastive learning 기본 (InfoNCE 손실)
  - D2 CLIP 아키텍처: image encoder + text encoder
  - D3 CLIP 논문 ("Learning Transferable Visual Models...") 정독
  - D4 `open_clip`으로 zero-shot 분류 해보기
  - D5 이미지↔텍스트 유사도 검색 미니 실습
- **Week 18 — CLIP 이후 개선들**
  - D1 SigLIP: sigmoid 손실이 바꾼 것
  - D2 BLIP: captioning + matching 멀티태스크
  - D3 ALIGN, EVA-CLIP 간단 비교
  - D4 대조학습의 한계 — 왜 "이해"가 아닌 "매칭"인가
  - D5 한 장 요약: 대조학습 계열 VLM 계보

---

## Phase 8. 현대 VLM — LLaVA 계열과 그 너머 (3주 / Week 19–21)

"이미지를 LLM에 넣는다"가 실제로 어떻게 되는지.

- **Week 19 — 구조 이해**
  - D1 VLM의 3요소: vision encoder + projector + LLM
  - D2 LLaVA 아키텍처 분해 (왜 단순 MLP projector로 충분한가)
  - D3 Q-Former (BLIP-2) 접근과 차이점
  - D4 Flamingo의 cross-attention 방식
  - D5 세 가지 접근을 한 표로
- **Week 20 — 학습 레시피**
  - D1 사전학습 vs instruction tuning 2단계
  - D2 Vision-language instruction 데이터 형식
  - D3 Frozen LLM + trainable projector 전략
  - D4 LoRA/QLoRA 개념 (VLM 파인튜닝에서 필수)
  - D5 LLaVA 논문 핵심 섹션 정독
- **Week 21 — 최신 흐름 훑기**
  - D1 Qwen-VL, InternVL 계열 특징
  - D2 고해상도 처리: AnyRes, tiling 전략
  - D3 동적 해상도 & multi-image 입력
  - D4 비디오 VLM 확장 개념
  - D5 "내가 VLM을 만든다면" 설계 노트 작성

---

## Phase 9. 캡스톤 — 작은 VLM 굴려보기 (2주 / Week 22–23)

- **Week 22**
  - D1 프로젝트 정의: pretrained ViT + 작은 LLM + MLP projector
  - D2 데이터셋 선택 (COCO Captions 일부 등)
  - D3 forward pass 조립, shape 디버깅
  - D4 projector만 학습하는 루프 작성
  - D5 첫 학습 — 손실 내려가는지 확인
- **Week 23**
  - D1 생성 결과 정성 평가
  - D2 간단한 VQA 예시로 테스트
  - D3 실패 사례 분석
  - D4 README + 모델카드 작성
  - D5 회고: 남은 궁금증 → 다음 학습 주제로

---

## 매일 30분 루틴

- **5분** — 어제 TIL 다시 읽기
- **20분** — 오늘 주제 (영상/논문 문단/코드 중 택1)
- **5분** — `til/YYYY-MM-DD.md`에 3줄 요약 + 질문 1개

## 레포 구조 제안

```
Deep-Learning-TIL/
├── CURRICULUM.md        # 이 문서
├── til/                 # 매일 기록
├── notebooks/           # 실습 노트북 (Phase별 폴더)
├── papers/              # 논문 요약 마크다운
└── projects/
    ├── nanogpt/         # Week 10 산출물
    └── tiny-vlm/        # Week 21–22 캡스톤
```

## 필수 레퍼런스

- **영상**: Karpathy "Zero to Hero" 시리즈 (특히 makemore, nanoGPT)
- **논문**: Attention Is All You Need / ViT / CLIP / LLaVA
- **코드**: `nanoGPT`, `open_clip`, `LLaVA` 공식 레포
- **도구**: HuggingFace `transformers`, `datasets`, `timm`

---

## 막힐 때 규칙

논문 수식이나 코드에서 막히면 넘어가지 말고 `til/`에 **"오늘 막힌 것"** 으로 남겨두기.
다음날 같이 풀어나가자 — 이 레포에서 내가 옆에 있을게.
