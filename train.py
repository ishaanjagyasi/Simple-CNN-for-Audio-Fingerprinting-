import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

from model import AudioFingerprintCNN
from dataset import get_dataloaders

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for triplet learning
    Encourages anchor-positive similarity and anchor-negative dissimilarity
    """
    
    def __init__(self, margin=16):
        """
        Args:
            margin: Minimum Hamming distance for negatives (in bits)
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def hamming_distance(self, x1, x2):
        """
        Compute Hamming distance between binary vectors
        For continuous values, use L1 distance as approximation
        """
        return torch.mean(torch.abs(x1 - x2), dim=1) * x1.shape[1]
    
    def forward(self, anchor, positive, negative):
        """
        Compute contrastive loss
        
        Args:
            anchor: Anchor embeddings (batch, 32)
            positive: Positive embeddings (batch, 32)
            negative: Negative embeddings (batch, 32)
        
        Returns:
            loss: Scalar loss value
        """
        # Distance between anchor and positive (should be small)
        pos_distance = self.hamming_distance(anchor, positive)
        
        # Distance between anchor and negative (should be large)
        neg_distance = self.hamming_distance(anchor, negative)
        
        # Positive loss: penalize large anchor-positive distance
        positive_loss = torch.mean(pos_distance)
        
        # Negative loss: penalize small anchor-negative distance (with margin)
        negative_loss = torch.mean(F.relu(self.margin - neg_distance))
        
        # Total loss
        loss = positive_loss + negative_loss
        
        return loss, positive_loss, negative_loss


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_pos_loss = 0
    total_neg_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (anchor, positive, negative) in enumerate(pbar):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        anchor_emb = model(anchor, binarize=False)
        positive_emb = model(positive, binarize=False)
        negative_emb = model(negative, binarize=False)
        
        # Compute loss
        loss, pos_loss, neg_loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_pos_loss += pos_loss.item()
        total_neg_loss += neg_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pos': f'{pos_loss.item():.4f}',
            'neg': f'{neg_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_pos_loss = total_pos_loss / len(train_loader)
    avg_neg_loss = total_neg_loss / len(train_loader)
    
    return avg_loss, avg_pos_loss, avg_neg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_pos_loss = 0
    total_neg_loss = 0
    
    with torch.no_grad():
        for anchor, positive, negative in tqdm(val_loader, desc="Validation"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            anchor_emb = model(anchor, binarize=False)
            positive_emb = model(positive, binarize=False)
            negative_emb = model(negative, binarize=False)
            
            # Compute loss
            loss, pos_loss, neg_loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_neg_loss += neg_loss.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_pos_loss = total_pos_loss / len(val_loader)
    avg_neg_loss = total_neg_loss / len(val_loader)
    
    return avg_loss, avg_pos_loss, avg_neg_loss


def train(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = AudioFingerprintCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Resumed from epoch {checkpoint['epoch']}, best val_loss: {best_val_loss:.4f}")
        else:
            print(f"Checkpoint {args.resume} not found, starting from scratch")
    
    # Training loop
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\nStarting training from epoch {start_epoch+1} to {args.epochs}...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_pos, train_neg = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_pos, val_neg = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} (pos: {train_pos:.4f}, neg: {train_neg:.4f})")
        print(f"Val Loss:   {val_loss:.4f} (pos: {val_pos:.4f}, neg: {val_neg:.4f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_dir / 'best_model.pth')
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Audio Fingerprinting CNN')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, default='./preprocessed/train',
                        help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='./preprocessed/val',
                        help='Path to validation data')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--margin', type=int, default=16,
                        help='Margin for contrastive loss (bits)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., ./checkpoints/best_model.pth)')
    
    args = parser.parse_args()
    
    train(args)
