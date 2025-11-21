import os
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class RandomDataset(Dataset):
    """Dataset that generates random data on the fly"""
    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image: (3, 32, 32)
        image = torch.randn(3, 32, 32)
        # Generate random label: 0-9
        label = torch.randint(0, 10, (1,)).item()
        return image, label


class SimpleCNN(nn.Module):
    """Basic CNN model for classification"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_network_interface():
    """Try to detect the appropriate network interface"""
    try:
        # Get hostname
        hostname = socket.gethostname()

        # Try common interfaces in order of preference
        # InfiniBand interfaces (common in HPC)
        ib_interfaces = ['ib0', 'ib1', 'mlx5_0', 'mlx5_1']
        # Ethernet interfaces
        eth_interfaces = ['eth0', 'eth1', 'eno1', 'enp0s31f6']

        all_interfaces = ib_interfaces + eth_interfaces

        # Try to use the first interface that exists
        import subprocess
        result = subprocess.run(['ip', 'addr', 'show'],
                              capture_output=True, text=True)

        for iface in all_interfaces:
            if iface in result.stdout:
                return iface

        # Fallback
        return 'eth0'
    except:
        return 'eth0'


def setup_distributed():
    """Initialize the distributed environment with proper NCCL configuration"""

    # Configure NCCL environment variables
    if 'NCCL_SOCKET_IFNAME' not in os.environ:
        network_interface = get_network_interface()
        os.environ['NCCL_SOCKET_IFNAME'] = network_interface
        print(f"Setting NCCL_SOCKET_IFNAME to: {network_interface}")

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, local_rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)

        if batch_idx % 10 == 0 and local_rank == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / total_samples
    return avg_loss


def main():
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    num_samples = 10000

    # Setup distributed training
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()

    if local_rank == 0:
        print(f"Hostname: {socket.gethostname()}")
        print(f"Training on {world_size} GPUs")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Effective batch size: {batch_size * world_size}")
        print(f"NCCL_SOCKET_IFNAME: {os.environ.get('NCCL_SOCKET_IFNAME', 'not set')}")

    # Create model and move to device
    model = SimpleCNN(num_classes=10).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])

    # Create dataset and sampler
    dataset = RandomDataset(size=(3, 32, 32), num_samples=num_samples)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    if local_rank == 0:
        print("\nStarting training...")

    for epoch in range(1, num_epochs + 1):
        # Set epoch for sampler to ensure proper shuffling
        sampler.set_epoch(epoch)

        # Train one epoch
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch, local_rank)

        if local_rank == 0:
            print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.6f}\n')

    # Save model (only on rank 0)
    if local_rank == 0:
        torch.save(model.module.state_dict(), 'ddp_cnn_model.pth')
        print("Model saved to ddp_cnn_model.pth")

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()