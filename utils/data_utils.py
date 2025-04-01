import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_mnist(batch_size=128, val_ratio=0.1, random_seed=42):
    """
    MNIST 데이터셋을 로드하고 학습/검증/테스트 데이터 로더를 반환
    
    Args:
        batch_size (int): 배치 크기
        val_ratio (float): 검증 데이터 비율
        random_seed (int): 랜덤 시드
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 데이터 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 학습 데이터 로드
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # 학습/검증 분할
    if val_ratio > 0:
        val_size = int(len(train_dataset) * val_ratio)
        train_size = len(train_dataset) - val_size
        
        torch.manual_seed(random_seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 테스트 데이터 로드
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def load_cifar10(batch_size=128, val_ratio=0.1, random_seed=42):
    """
    CIFAR-10 데이터셋을 로드하고 학습/검증/테스트 데이터 로더를 반환
    
    Args:
        batch_size (int): 배치 크기
        val_ratio (float): 검증 데이터 비율
        random_seed (int): 랜덤 시드
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 데이터 변환
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 학습 데이터 로드
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    # 학습/검증 분할
    if val_ratio > 0:
        val_size = int(len(train_dataset) * val_ratio)
        train_size = len(train_dataset) - val_size
        
        torch.manual_seed(random_seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # 검증 데이터는 테스트 변환 적용
        val_dataset.dataset.transform = transform_test
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 테스트 데이터 로드
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def load_custom_dataset(X, y, batch_size=128, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    사용자 정의 데이터셋을 로드하고 학습/검증/테스트 데이터 로더를 반환
    
    Args:
        X (numpy.ndarray): 입력 데이터
        y (numpy.ndarray): 타깃 데이터
        batch_size (int): 배치 크기
        val_ratio (float): 검증 데이터 비율
        test_ratio (float): 테스트 데이터 비율
        random_seed (int): 랜덤 시드
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # NumPy 배열을 PyTorch 텐서로 변환
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # 데이터셋 생성
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # 학습/검증/테스트 분할
    test_size = int(len(dataset) * test_ratio)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - test_size - val_size
    
    torch.manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader 