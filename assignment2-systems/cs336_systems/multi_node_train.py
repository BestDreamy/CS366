import torch
import torch.distributed as dist
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import time
from torch.multiprocessing.spawn import spawn
import numpy as np
import random

class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
        super().__init__()
        
        # Create a list of layer sizes
        layer_sizes = [input_size] + hidden_sizes + [num_classes]
        
        # Create the layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    
    def forward(self, x):
        # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = torch.flatten(x, 1)
        
        # Forward through all layers except the last one with ReLU activation
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        # Last layer without activation (for use with cross-entropy loss)
        x = self.layers[-1](x)
        
        return F.log_softmax(x, dim=1)
    
def setup(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29511'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ddp_train(rank, world_size, backend):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    setup(rank, world_size, backend)
    
    gpu_id = rank  # rank 0 -> GPU 0, rank 1 -> GPU 1
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Rank {rank} using GPU: {device}")
    else:
        device = torch.device("cpu")
        print(f"CUDA not available, using CPU for rank {rank}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    
    total_samples = len(dataset)
    samples_per_process = total_samples // world_size
    start_idx = rank * samples_per_process
    end_idx = start_idx + samples_per_process
    
    subset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True, generator=generator)
    
    print(f"Rank {rank} processing samples {start_idx} to {end_idx-1} ({samples_per_process} samples)")
    print(f"Rank {rank} effective batch size: 64, total distributed batch size: {64 * world_size}")
    
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    for group in optimizer.param_groups:
        for param in group['params']:
            if param in optimizer.state:
                for key, value in optimizer.state[param].items():
                    if torch.is_tensor(value):
                        dist.broadcast(value, src=0)
    
    model.train()
    
    for epoch in range(1, 3):  # Train for 2 epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
            
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f'Rank {rank}, Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    if rank == 0:
        torch.save(model.state_dict(), "mnist_simple_ddp.pt")
        print("Model saved!")
            
    cleanup()

def main():
    world_size = 2
    backend = 'nccl'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    data_size = len(dataset)

    if data_size % world_size != 0:
        print(f"Warning: Data size {data_size} is not divisible by world size {world_size}")
        print(f"Some samples will be ignored to ensure equal distribution")
    
    print(f"Starting distributed training with {world_size} processes")
    print(f"Each process will handle {data_size // world_size} samples")
    print(f"Total samples: {data_size}")
    
    spawn(ddp_train, args=(world_size, backend), nprocs=world_size, join=True)
    
    print("Distributed training completed!")

if __name__ == '__main__':
    main()