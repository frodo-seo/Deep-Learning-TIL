# 다음 세션 시작점

**현재 위치**: Week 1 · Day 1 — 벡터와 내적의 기하적 의미
**모드**: 대화형 (개념 → 질문 → 답하며 체화)
**예상 시간**: 30분

## 이어서 할 일

1. **이전 턴에서 받은 질문 Q1 먼저 답하기**
   - 벡터 `v = [4, 3]` 을 화살표로 그리면 어디로 가? 이 화살표의 길이는?
2. 답한 뒤 **내적(dot product)** 으로 진행:
   - 대수적 정의: `a·b = a₁b₁ + a₂b₂ + …`
   - 기하적 정의: `a·b = |a|·|b|·cos θ`
   - 두 정의가 **같다**는 것 → "얼마나 같은 방향을 보는가"
   - Transformer Attention의 `Query · Key` 점수로 연결
3. D1 마무리 — `til/YYYY-MM-DD.md`에 3줄 요약 + 궁금한 것 1개

## 다음 세션 시작 한 줄

> "이어서 Week 1 D1 갈게" 라고만 말해줘. Claude가 바로 Q1 물으면서 재개함.

## 참고 자료 (이번 주)

- 3Blue1Brown *Essence of Linear Algebra* Ep. 1–4 (30분 세션에 영상 1개씩)
- NumPy: `np.array`, `.shape`, `@`, broadcasting

---

## Week 1 전체 일정 (상기용)

- D1 ← **여기**
- D2 행렬곱 = 선형변환 합성 + NumPy `@`
- D3 브로드캐스팅·슬라이싱·reshape 집중
- D4 전치·역행렬·랭크 직관
- D5 `(B, T, C)` 차원 읽기 + Attention 수식 차원 분해

더 자세한 건 루트의 `CURRICULUM.md`.
