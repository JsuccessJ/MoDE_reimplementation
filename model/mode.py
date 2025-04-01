import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from .moe import MoE, Expert, GatingNetwork

class MoDE(nn.Module):
    """
    Mixture of Distilled Experts (MoDE) 모델
    MoE를 기반으로 상호 지식 증류(mutual knowledge distillation)를 적용한 모델
    """
    def __init__(self, num_experts, input_dim, hidden_dim, output_dim, 
                 sparse_gate=False, top_k=None, distillation_temp=2.0, alpha=0.5):
        super(MoDE, self).__init__()
        
        # 기본 MoE 모델 생성
        self.moe = MoE(num_experts, input_dim, hidden_dim, output_dim, sparse_gate, top_k)
        
        # 지식 증류를 위한 하이퍼파라미터
        self.distillation_temp = distillation_temp  # 증류 온도
        self.alpha = alpha  # 지식 증류와 일반 손실 사이의 가중치 조절 파라미터
        
        # 손실 함수 및 최적화기
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        
    def forward(self, x):
        """순방향 전파"""
        return self.moe(x)
    
    def get_expert_outputs(self, x):
        """각 전문가 모델의 출력을 개별적으로 계산"""
        expert_outputs = []
        for expert in self.moe.experts:
            expert_outputs.append(expert(x))
        return expert_outputs
        
    def configure_optimizer(self, lr=0.001):
        """최적화 알고리즘 설정"""
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def compute_distillation_loss(self, expert_outputs, gates):
        """
        전문가 간 지식 증류를 위한 손실 계산
        각 전문가는 다른 전문가의 출력에서 배우게 됨
        """
        distillation_loss = 0
        num_experts = len(expert_outputs)
        
        # 각 전문가 쌍에 대한 KL 발산 계산
        for i in range(num_experts):
            for j in range(num_experts):
                if i != j:  # 다른 전문가끼리만
                    # 소프트 타깃 생성 (로짓을 소프트맥스 확률로 변환)
                    soft_target = F.softmax(expert_outputs[j] / self.distillation_temp, dim=1)
                    # 학생 모델의 로그 소프트맥스
                    log_pred = F.log_softmax(expert_outputs[i] / self.distillation_temp, dim=1)
                    
                    # KL 발산 계산 (소프트 타깃과 예측 사이)
                    kl_div = F.kl_div(log_pred, soft_target, reduction='batchmean')
                    
                    # 게이팅 가중치에 따라 KL 발산 가중 적용
                    weighted_kl = kl_div * gates[:, i].mean() * gates[:, j].mean()
                    distillation_loss += weighted_kl
        
        # 전문가 쌍의 수로 정규화
        return distillation_loss / (num_experts * (num_experts - 1))
    
    def train_step(self, x, y):
        """단일 학습 단계 (지식 증류 포함)"""
        self.optimizer.zero_grad()
        
        # 게이팅 값과 전문가 출력 계산
        gates = self.moe.get_gate_values(x)
        expert_outputs = self.get_expert_outputs(x)
        
        # MoE의 최종 출력
        moe_output = self.forward(x)
        
        # 일반 분류 손실
        ce_loss = self.loss_fn(moe_output, y)
        
        # 지식 증류 손실
        distillation_loss = self.compute_distillation_loss(expert_outputs, gates)
        
        # 총 손실 = 일반 손실 + 증류 손실의 가중 합
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * self.distillation_temp**2 * distillation_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'distillation_loss': distillation_loss.item()
        }
    
    def train_model(self, train_loader, val_loader=None, epochs=10, lr=0.001):
        """전체 모델 학습"""
        if self.optimizer is None:
            self.configure_optimizer(lr)
            
        self.train()
        history = {
            'train_loss': [], 
            'train_ce_loss': [], 
            'train_dist_loss': [],
            'val_loss': [], 
            'val_acc': []
        }
        
        for epoch in range(epochs):
            train_total_loss = 0
            train_ce_loss = 0
            train_dist_loss = 0
            
            for x_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                losses = self.train_step(x_batch, y_batch)
                train_total_loss += losses['total_loss']
                train_ce_loss += losses['ce_loss']
                train_dist_loss += losses['distillation_loss']
                
            # 에폭당 평균 손실
            avg_train_loss = train_total_loss / len(train_loader)
            avg_ce_loss = train_ce_loss / len(train_loader)
            avg_dist_loss = train_dist_loss / len(train_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['train_ce_loss'].append(avg_ce_loss)
            history['train_dist_loss'].append(avg_dist_loss)
            
            # 검증 수행 (있는 경우)
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
                      f'CE Loss: {avg_ce_loss:.4f}, Dist Loss: {avg_dist_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
                      f'CE Loss: {avg_ce_loss:.4f}, Dist Loss: {avg_dist_loss:.4f}')
                
        return history
    
    def evaluate(self, data_loader):
        """검증 또는 테스트 데이터에 대한 평가"""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                outputs = self(x_batch)
                loss = self.loss_fn(outputs, y_batch)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
        self.train()
        return total_loss / len(data_loader), correct / total
    
    def predict(self, x):
        """예측 수행"""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted 