import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 루트 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_mnist

def visualize_mnist_samples(num_samples=25):
    """MNIST 데이터셋에서 무작위 샘플을 선택하여 시각화"""
    
    # MNIST 데이터셋 로드
    train_loader, _, _ = load_mnist(batch_size=100, val_ratio=0)
    
    # 데이터 배치 하나 가져오기
    for images, labels in train_loader:
        break
    
    # 표시할 샘플 수 확인
    num_samples = min(num_samples, len(images))
    
    # 그리드 크기 계산 (예: 5x5)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # 시각화
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(images[i][0], cmap='gray')  # MNIST는 단일 채널(흑백)
        plt.title(f'레이블: {labels[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png')  # 이미지 저장
    plt.show()

def visualize_specific_digits():
    """각 숫자(0-9)에 대한 예시 이미지 시각화"""
    
    # MNIST 데이터셋 로드
    train_loader, _, _ = load_mnist(batch_size=10000, val_ratio=0)
    
    # 첫 배치에서 모든 학습 데이터 가져오기
    for images, labels in train_loader:
        break
    
    # 각 숫자별 이미지 찾기
    digit_indices = {}
    for i in range(10):
        # 각 숫자에 대한 첫 번째 인덱스 찾기
        indices = (labels == i).nonzero()
        if len(indices) > 0:
            digit_indices[i] = indices[0].item()
    
    # 시각화
    plt.figure(figsize=(12, 2.5))
    for i in range(10):
        if i in digit_indices:
            plt.subplot(1, 10, i+1)
            plt.imshow(images[digit_indices[i]][0], cmap='gray')
            plt.title(f'숫자: {i}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_digits_0_to_9.png')  # 이미지 저장
    plt.show()

if __name__ == "__main__":
    print("MNIST 데이터셋 샘플 시각화")
    visualize_mnist_samples(25)
    print("각 숫자(0-9)에 대한 예시 이미지")
    visualize_specific_digits() 