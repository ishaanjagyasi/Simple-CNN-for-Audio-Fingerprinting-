import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioFingerprintCNN(nn.Module):
    """
    Lightweight CNN for audio fingerprinting
    Produces 32-bit binary fingerprints from spectrogram patches
    """
    
    def __init__(self):
        super(AudioFingerprintCNN, self).__init__()
        
        # Conv Block 1: 1 -> 16 channels, 5x5 filters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Block 2: 16 -> 32 channels, 3x3 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Block 3: 32 -> 64 channels, 3x3 filters
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # --- ARCHITECTURE FIX ---
        # Removed Global Average Pooling which was destroying spatial information.
        # Calculated Flattened Size:
        # Input: (1, 128, 64)
        # After Pool1 (2x2) -> (16, 64, 32)
        # After Pool2 (2x2) -> (32, 32, 16)
        # After Pool3 (2x2) -> (64, 16, 8)
        # Flattened = 64 channels * 16 freq * 8 time = 8192
        
        self.fc = nn.Linear(8192, 32)
        
    def forward(self, x, binarize=False):
        """
        Forward pass
        
        Args:
            x: Input spectrogram patches (batch, 1, 128, 64)
            binarize: If True, apply sign function for binary output
        
        Returns:
            Fingerprint embeddings (batch, 32)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # --- ARCHITECTURE FIX ---
        # Flatten the output while preserving the batch dimension
        x = x.view(x.size(0), -1)  # Flatten to (batch, 8192)
        
        # Fully connected to 32 dimensions
        x = self.fc(x)
        
        # Apply tanh to push values towards -1 or +1
        x = torch.tanh(x)
        
        # Binarization using sign function
        if binarize:
            x = self.sign_with_gradient(x)
        
        return x
    
    @staticmethod
    def sign_with_gradient(x):
        """
        Sign function with straight-through estimator
        Forward: sign(x)
        Backward: gradient = 1 (identity)
        """
        # Forward pass: binarize
        binary = torch.sign(x)
        
        # Backward pass: use straight-through estimator
        # Gradient flows as if sign was identity function
        return binary - x.detach() + x


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = AudioFingerprintCNN()
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 64)
    
    # Without binarization
    output = model(dummy_input, binarize=False)
    print(f"Output shape (continuous): {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # With binarization
    output_binary = model(dummy_input, binarize=True)
    print(f"Output shape (binary): {output_binary.shape}")
    print(f"Unique values: {torch.unique(output_binary).tolist()}")