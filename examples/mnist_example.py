import torch
import sys
import os

# 루트 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MoE, MoDE
from utils import load_mnist, plot_loss_curves, plot_accuracy, visualize_expert_weights

def main():
    # MNIST 데이터셋 로드
    train_loader, val_loader, test_loader = load_mnist(batch_size=64, val_ratio=0.1)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # MoE 모델 정의
    moe_model = MoE(
        num_experts=3,
        input_dim=28*28,
        hidden_dim=128,
        output_dim=10
    ).to(device)
    
    # MoDE 모델 정의
    mode_model = MoDE(
        num_experts=3,
        input_dim=28*28,
        hidden_dim=128,
        output_dim=10,
        distillation_temp=2.0,
        alpha=0.5
    ).to(device)
    
    # 모델 학습
    print("\nTraining MoE model...")
    moe_history = train_model(moe_model, train_loader, val_loader, device, epochs=5, lr=0.001)
    
    print("\nTraining MoDE model...")
    mode_history = train_model(mode_model, train_loader, val_loader, device, epochs=5, lr=0.001)
    
    # 테스트 데이터로 평가
    print("\nEvaluating models on test data...")
    moe_test_loss, moe_test_acc = evaluate_model(moe_model, test_loader, device)
    mode_test_loss, mode_test_acc = evaluate_model(mode_model, test_loader, device)
    
    print(f"MoE Test Accuracy: {moe_test_acc:.4f}")
    print(f"MoDE Test Accuracy: {mode_test_acc:.4f}")
    
    # 성능 비교 시각화
    plot_accuracy(
        [moe_test_acc], 
        [mode_test_acc], 
        labels=['MNIST']
    )
    
    # 손실 곡선 시각화
    print("\nPlotting loss curves...")
    plot_loss_curves(moe_history)
    plot_loss_curves(mode_history)
    
    # 전문가 가중치 시각화
    print("\nVisualizing expert weights...")
    # 테스트 데이터의 배치 하나 샘플링
    for x_batch, _ in test_loader:
        x_batch = x_batch.view(x_batch.size(0), -1).to(device)
        break
    
    visualize_expert_weights(moe_model, x_batch)
    visualize_expert_weights(mode_model, x_batch)

def train_model(model, train_loader, val_loader, device, epochs=5, lr=0.001):
    """모델 학습 함수"""
    # 데이터를 디바이스로 이동하는 함수
    def to_device(data, target):
        data = data.view(data.size(0), -1).to(device)  # MNIST 이미지를 벡터로 변환
        target = target.to(device)
        return data, target
    
    # 학습 루프
    if hasattr(model, 'train_model'):
        # MoDE 또는 MoE 클래스의 학습 메서드 사용
        history = {}
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = to_device(x_batch, y_batch)
            sample_batch = (x_batch, y_batch)
            break
        
        # 데이터 로더를 변환된 형식으로 래핑
        class DeviceLoader:
            def __init__(self, loader, device):
                self.loader = loader
                self.device = device
            
            def __iter__(self):
                for data, target in self.loader:
                    yield to_device(data, target)
            
            def __len__(self):
                return len(self.loader)
        
        train_dev_loader = DeviceLoader(train_loader, device)
        val_dev_loader = DeviceLoader(val_loader, device) if val_loader else None
        
        # 모델 학습 메서드 호출
        history = model.train_model(train_dev_loader, val_dev_loader, epochs=epochs, lr=lr)
    
    return history

def evaluate_model(model, test_loader, device):
    """모델 평가 함수"""
    # 데이터를 디바이스로 이동
    def to_device(data, target):
        data = data.view(data.size(0), -1).to(device)  # MNIST 이미지를 벡터로 변환
        target = target.to(device)
        return data, target
    
    # 평가 로더 래핑
    class DeviceLoader:
        def __init__(self, loader, device):
            self.loader = loader
            self.device = device
        
        def __iter__(self):
            for data, target in self.loader:
                yield to_device(data, target)
        
        def __len__(self):
            return len(self.loader)
    
    test_dev_loader = DeviceLoader(test_loader, device)
    
    # 모델 평가
    loss, acc = model.evaluate(test_dev_loader)
    return loss, acc

if __name__ == "__main__":
    main() 