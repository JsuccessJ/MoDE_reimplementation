import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def plot_loss_curves(history, figsize=(12, 5)):
    """
    학습 손실 및 검증 손실 곡선을 시각화
    
    Args:
        history (dict): 학습 히스토리
        figsize (tuple): 그림 크기
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 손실 곡선
    ax1.plot(history['train_loss'], label='Train Loss')
    if 'train_ce_loss' in history:
        ax1.plot(history['train_ce_loss'], label='CE Loss')
    if 'train_dist_loss' in history:
        ax1.plot(history['train_dist_loss'], label='Distillation Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(history['val_loss'], label='Validation Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 정확도 곡선 (있는 경우)
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_accuracy(moe_acc, mode_acc, labels=None, figsize=(10, 6)):
    """
    MoE와 MoDE의 정확도를 비교하는 막대 그래프 시각화
    
    Args:
        moe_acc (list): MoE 모델들의 정확도
        mode_acc (list): MoDE 모델들의 정확도
        labels (list): 실험 레이블
        figsize (tuple): 그림 크기
    """
    if labels is None:
        labels = [f'Exp {i+1}' for i in range(len(moe_acc))]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width/2, moe_acc, width, label='MoE')
    rects2 = ax.bar(x + width/2, mode_acc, width, label='MoDE')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('MoE vs MoDE Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # 각 막대 위에 값 표시
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.show()

def visualize_expert_weights(model, inputs, cmap='viridis', figsize=(12, 5)):
    """
    입력에 대한 각 전문가의 가중치를 시각화
    
    Args:
        model: MoE 또는 MoDE 모델
        inputs (torch.Tensor): 입력 데이터
        cmap (str): 컬러맵
        figsize (tuple): 그림 크기
    """
    # 게이팅 네트워크 출력 계산
    if hasattr(model, 'moe'):  # MoDE 모델인 경우
        gates = model.moe.get_gate_values(inputs)
    else:  # MoE 모델인 경우
        gates = model.get_gate_values(inputs)
    
    # 배치 차원에 대해 평균
    avg_gates = gates.mean(dim=0).detach().cpu().numpy()
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 막대 그래프
    ax1.bar(range(len(avg_gates)), avg_gates)
    ax1.set_xlabel('Expert Index')
    ax1.set_ylabel('Average Weight')
    ax1.set_title('Average Expert Weights')
    ax1.set_xticks(range(len(avg_gates)))
    ax1.grid(True, alpha=0.3)
    
    # 히트맵 (샘플별 게이팅 가중치)
    im = ax2.imshow(gates.detach().cpu().numpy()[:min(100, gates.shape[0])], 
                   aspect='auto', cmap=cmap)
    ax2.set_xlabel('Expert Index')
    ax2.set_ylabel('Sample Index (first 100)')
    ax2.set_title('Expert Weights per Sample')
    ax2.set_xticks(range(gates.shape[1]))
    fig.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return avg_gates 