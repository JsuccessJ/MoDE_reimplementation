# MoDE (Mixture of Distilled Experts) 구현

이 프로젝트는 MoDE(Mixture of Distilled Experts) 모델의 구현입니다. MoDE는 전문가 모델(Expert Models)을 사용하는 MoE(Mixture of Experts)의 개선된 버전으로, 상호 지식 증류(mutual knowledge distillation)를 통해 모델의 일반화 능력을 향상시킵니다.

## 주요 특징

- MoE(Mixture of Experts) 기본 구현
- MoDE(Mixture of Distilled Experts) 구현
- 컴퓨터 비전 데이터셋에 대한 실험 지원
- 모델 해석 기능

## 설치 방법

필요한 패키지를 설치하려면:

```bash
pip install -r requirements.txt
```

## 사용 방법

```python
from mode import MoDE, MoE

# 모델 초기화
mode_model = MoDE(num_experts=2, input_dim=784, hidden_dim=256, output_dim=10)

# 학습
mode_model.train(train_loader, epochs=10)

# 평가
accuracy = mode_model.evaluate(test_loader)
```

## 프로젝트 구조

- `model/`: MoDE 및 MoE 모델 구현
- `utils/`: 유틸리티 함수
- `experiments/`: 실험 스크립트
- `examples/`: 사용 예제

## 참고 문헌

본 구현은 MoDE 논문을 기반으로 합니다. 