import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm

# 루트 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MoE, MoDE
from utils import load_mnist, visualize_expert_weights

class FeatureImportanceAnalyzer:
    """MoE 및 MoDE 모델의 전문가별 특징 중요도 분석 클래스"""
    
    def __init__(self, model):
        self.model = model
        self.is_mode = isinstance(model, MoDE)
    
    def analyze_feature_importance(self, data_loader):
        """
        선형 회귀를 사용하여 각 전문가의 입력 특징 중요도를 분석
        
        Args:
            data_loader: 분석에 사용할 데이터 로더
        
        Returns:
            dict: 각 전문가별 특징 중요도
        """
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 입력 차원 가져오기 (첫 번째 배치의 첫 번째 샘플 기준)
        for x_batch, _ in data_loader:
            x_batch = x_batch.view(x_batch.size(0), -1)
            input_dim = x_batch.shape[1]
            break
        
        # 각 전문가별 특징 중요도를 저장할 딕셔너리
        feature_importance = {}
        
        # MoDE인 경우 전문가 모델을 가져옴
        experts = self.model.moe.experts if self.is_mode else self.model.experts
        num_experts = len(experts)
        
        for expert_idx, expert in enumerate(experts):
            print(f"전문가 {expert_idx+1} 분석 중...")
            
            # 최종 레이어 출력 차원
            output_dim = expert.fc3.out_features
            
            # 각 출력 뉴런에 대한 특징 중요도 분석
            importance_per_neuron = []
            
            # 각 출력 뉴런에 대해
            for neuron_idx in range(output_dim):
                # 각 출력 뉴런의 activations과 입력 특징 사이의 관계를 분석
                inputs_all = []
                outputs_all = []
                
                # 배치 데이터를 수집
                with torch.no_grad():
                    for x_batch, _ in tqdm(data_loader, desc=f"출력 뉴런 {neuron_idx+1}/{output_dim} 분석"):
                        x_batch = x_batch.view(x_batch.size(0), -1)
                        
                        # 전문가의 출력 계산
                        expert_output = expert(x_batch)
                        
                        # 특정 뉴런의 출력 가져오기
                        neuron_output = expert_output[:, neuron_idx]
                        
                        inputs_all.append(x_batch.cpu().numpy())
                        outputs_all.append(neuron_output.cpu().numpy())
                
                # 데이터 결합
                inputs_all = np.vstack(inputs_all)
                outputs_all = np.concatenate(outputs_all)
                
                # 선형 회귀 모델 훈련 (최소 제곱법 사용)
                # X^T * X * w = X^T * y
                XTX = inputs_all.T @ inputs_all
                XTy = inputs_all.T @ outputs_all
                
                # 정규화 추가 (안정성을 위해)
                reg_lambda = 1e-6
                weights = np.linalg.solve(XTX + reg_lambda * np.eye(input_dim), XTy)
                
                importance_per_neuron.append(weights)
            
            # 모든 출력 뉴런에 대한 중요도 평균
            avg_importance = np.abs(np.vstack(importance_per_neuron)).mean(axis=0)
            feature_importance[f'expert_{expert_idx}'] = avg_importance
        
        return feature_importance
    
    def compare_feature_rankings(self, moe_importance, mode_importance):
        """
        MoE와 MoDE 간의 특징 중요도 순위 비교
        
        Args:
            moe_importance (dict): MoE 모델의 특징 중요도
            mode_importance (dict): MoDE 모델의 특징 중요도
        
        Returns:
            dict: 비교 결과
        """
        comparison = {}
        
        for expert_idx in range(len(moe_importance)):
            expert_key = f'expert_{expert_idx}'
            
            # 특징 중요도 순위 계산
            moe_ranks = np.argsort(moe_importance[expert_key])[::-1]
            mode_ranks = np.argsort(mode_importance[expert_key])[::-1]
            
            # 순위 변화 분석
            rank_changes = {}
            for feature_idx in range(len(moe_ranks)):
                moe_rank = np.where(moe_ranks == feature_idx)[0][0]
                mode_rank = np.where(mode_ranks == feature_idx)[0][0]
                rank_changes[feature_idx] = moe_rank - mode_rank
            
            # 순위가 상승한 특징들
            improved_features = {idx: change for idx, change in rank_changes.items() if change > 0}
            
            comparison[expert_key] = {
                'rank_changes': rank_changes,
                'improved_features': improved_features,
                'num_improved': len(improved_features)
            }
        
        return comparison
    
    def visualize_feature_importance(self, feature_importance, num_top_features=20):
        """
        각 전문가의 상위 특징 중요도 시각화
        
        Args:
            feature_importance (dict): 특징 중요도 딕셔너리
            num_top_features (int): 표시할 상위 특징 수
        """
        num_experts = len(feature_importance)
        
        plt.figure(figsize=(15, 5 * num_experts))
        
        for i, (expert_key, importance) in enumerate(feature_importance.items()):
            # 상위 특징 선택
            top_indices = np.argsort(importance)[-num_top_features:]
            top_values = importance[top_indices]
            
            # 특징 인덱스를 문자열로 변환
            feature_names = [f'Feature {idx}' for idx in top_indices]
            
            plt.subplot(num_experts, 1, i+1)
            plt.barh(feature_names, top_values)
            plt.xlabel('중요도')
            plt.ylabel('특징')
            plt.title(f'{expert_key}의 상위 {num_top_features}개 특징')
            plt.tight_layout()
        
        plt.savefig('feature_importance.png')
        plt.show()
        
    def visualize_feature_rank_changes(self, comparison, expert_idx=0, num_features=20):
        """
        MoE와 MoDE 간의 특징 순위 변화 시각화
        
        Args:
            comparison (dict): 비교 결과
            expert_idx (int): 시각화할 전문가 인덱스
            num_features (int): 표시할 특징 수
        """
        expert_key = f'expert_{expert_idx}'
        rank_changes = comparison[expert_key]['rank_changes']
        
        # 순위 변화가 큰 순서대로 정렬
        sorted_features = sorted(rank_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:num_features]
        feature_indices = [f'Feature {idx}' for idx, _ in sorted_features]
        changes = [change for _, change in sorted_features]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(feature_indices, changes, color=['red' if x < 0 else 'green' for x in changes])
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('순위 변화 (양수: MoDE에서 순위 상승)')
        plt.ylabel('특징')
        plt.title(f'전문가 {expert_idx}의 MoE와 MoDE 간 특징 순위 변화')
        
        # 범례 추가
        plt.legend([bars[0], bars[-1 if changes[-1] < 0 else 0]], ['순위 하락', '순위 상승'])
        
        plt.tight_layout()
        plt.savefig(f'feature_rank_changes_expert_{expert_idx}.png')
        plt.show()

def main():
    # 데이터 로드
    batch_size = 1000
    train_loader, val_loader, test_loader = load_mnist(batch_size=batch_size)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 입력 차원 설정
    for x, _ in train_loader:
        input_dim = x.view(x.size(0), -1).size(1)  # 784 (28x28)
        break
    
    # 모델 설정
    hidden_dim = 128
    output_dim = 10
    num_experts = 2
    
    # MoE 모델 생성 및 간단한 훈련
    print("MoE 모델 훈련 중...")
    moe_model = MoE(num_experts, input_dim, hidden_dim, output_dim).to(device)
    
    # MoDE 모델 생성 및 간단한 훈련
    print("MoDE 모델 훈련 중...")
    mode_model = MoDE(
        num_experts, 
        input_dim, 
        hidden_dim, 
        output_dim, 
        distillation_temp=2.0, 
        alpha=0.01  # 논문에서 제시한 적절한 알파값
    ).to(device)
    
    # 간단한 훈련 데이터 전처리 및 장치 이동 함수
    def preprocess_batch(x, y):
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        return x, y
    
    # 학습 함수
    def train_model(model, epochs=5):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for x, y in tqdm(train_loader, desc=f'에폭 {epoch+1}/{epochs}'):
                x, y = preprocess_batch(x, y)
                
                optimizer.zero_grad()
                if isinstance(model, MoDE):
                    # MoDE 모델의 경우
                    gates = model.moe.get_gate_values(x)
                    expert_outputs = model.get_expert_outputs(x)
                    outputs = model(x)
                    
                    # 일반 손실
                    ce_loss = criterion(outputs, y)
                    
                    # 지식 증류 손실
                    distillation_loss = model.compute_distillation_loss(expert_outputs, gates)
                    
                    # 총 손실
                    loss = (1 - model.alpha) * ce_loss + model.alpha * model.distillation_temp**2 * distillation_loss
                else:
                    # MoE 모델의 경우
                    outputs = model(x)
                    loss = criterion(outputs, y)
                
                loss.backward()
                optimizer.step()
    
    # 모델 훈련
    train_model(moe_model)
    train_model(mode_model)
    
    # 시각화할 배치 준비
    for batch_x, _ in test_loader:
        sample_x = batch_x.view(batch_x.size(0), -1).to(device)
        break
    
    # 각 모델의 게이팅 네트워크 가중치 시각화
    print("MoE 모델의 게이팅 가중치 시각화...")
    moe_gates = visualize_expert_weights(moe_model, sample_x)
    
    print("MoDE 모델의 게이팅 가중치 시각화...")
    mode_gates = visualize_expert_weights(mode_model, sample_x)
    
    # 특징 중요도 분석
    print("특징 중요도 분석 중...")
    moe_analyzer = FeatureImportanceAnalyzer(moe_model)
    mode_analyzer = FeatureImportanceAnalyzer(mode_model)
    
    # 특징 중요도 계산 (참고: 계산 시간이 오래 걸릴 수 있음)
    print("MoE 모델의 특징 중요도 분석 중...")
    moe_importance = moe_analyzer.analyze_feature_importance(test_loader)
    
    print("MoDE 모델의 특징 중요도 분석 중...")
    mode_importance = mode_analyzer.analyze_feature_importance(test_loader)
    
    # 특징 중요도 시각화
    print("MoE 모델의 특징 중요도 시각화...")
    moe_analyzer.visualize_feature_importance(moe_importance)
    
    print("MoDE 모델의 특징 중요도 시각화...")
    mode_analyzer.visualize_feature_importance(mode_importance)
    
    # 특징 순위 변화 비교
    print("MoE와 MoDE 간의 특징 순위 변화 분석 중...")
    comparison = moe_analyzer.compare_feature_rankings(moe_importance, mode_importance)
    
    # 각 전문가에 대한 분석 결과 출력
    for expert_idx in range(num_experts):
        expert_key = f'expert_{expert_idx}'
        print(f"\n전문가 {expert_idx}의 분석 결과:")
        print(f"순위가 상승한 특징 수: {comparison[expert_key]['num_improved']}")
        
        # 상위 10개 순위 변화 출력
        top_changes = sorted(
            comparison[expert_key]['rank_changes'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        print("상위 10개 순위 상승 특징:")
        for feature_idx, change in top_changes:
            print(f"특징 {feature_idx}: 순위 변화 {change}")
    
    # 순위 변화 시각화
    for expert_idx in range(num_experts):
        print(f"\n전문가 {expert_idx}의 순위 변화 시각화...")
        moe_analyzer.visualize_feature_rank_changes(comparison, expert_idx)

if __name__ == "__main__":
    main() 