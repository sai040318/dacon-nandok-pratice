# dacon-nandok-pratice

본 레포지토리는 6명이 함께 진행하는 팀 프로젝트로,  
Baseline → 전처리 개선 → 경량 모델 → LoRA → 최종 앙상블  
순서로 발전시키는 구조입니다.

##  협업 규칙 (Git Workflow)
- `main` : 최종 제출 버전만 머지 (직접 push 금지)
- `develop` : 각자 작업물 통합 브랜치
- `feature/*` : 개인 작업 브랜치 (여기서만 자유롭게 수정)

브랜치 예시:
- feature/preprocess_sehun
- feature/preprocess_dahyoung
- feature/lightmodel_seunghwan
- feature/lightmodel_junbeom
- feature/lora_minseok
- feature/lora_taewon

## 작업 흐름 (Workflow)

- 각자 feature/개인브랜치에서 작업
- 기능이 어느 정도 완성되면 develop으로 PR(Pull Request)
- 팀원 1명 이상 리뷰 후 merge
- develop이 안정화되면 → main에 최종 merge

## Pull Request 규칙

- 제목: [preprocess] 규칙 기반 전처리 추가, [lora] LoRA 1차 실험
- 내용:
- 변경 내용
- baseline 대비 달라진 점
- 실험 결과(간단하게)
- 영향받는 모듈등등
