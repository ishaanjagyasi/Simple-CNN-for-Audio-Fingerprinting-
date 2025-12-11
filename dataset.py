import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from scipy.ndimage import gaussian_filter

class AudioFingerprintDataset(Dataset):
    """
    Dataset for audio fingerprinting with Hard Negative Mining
    Returns: (anchor, positive, negative) patches
    """
    
    def __init__(self, data_dir, patch_size=64, transform=True):
        """
        Args:
            data_dir: Directory containing preprocessed spectrograms
            patch_size: Number of time frames per patch (64)
            transform: Whether to apply augmentations to positive samples
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.transform = transform
        
        # Load all spectrogram files (filter out hidden files)
        self.spec_files = sorted([f for f in self.data_dir.glob('*_spec.npy') if not f.name.startswith('.')])
        print(f"Loaded {len(self.spec_files)} spectrograms from {data_dir}")
        
    def _load_spectrogram(self, idx):
        """Load spectrogram"""
        spec = np.load(self.spec_files[idx])
        return spec
    
    def _extract_random_patch(self, idx):
        """Extract one random patch from spectrogram"""
        spec = self._load_spectrogram(idx)
        n_freq, n_time = spec.shape
        
        # Check if spectrogram is long enough
        if n_time < self.patch_size:
            # Pad if too short
            pad_width = ((0, 0), (0, self.patch_size - n_time))
            spec = np.pad(spec, pad_width, mode='constant')
            n_time = self.patch_size
        
        # Randomly select start time
        max_start = n_time - self.patch_size
        start_t = random.randint(0, max_start)
        
        patch = spec[:, start_t:start_t+self.patch_size]
        return patch
    
    def _augment_patch(self, patch):
        """
        Apply all augmentations with fixed strengths
        """
        patch = patch.copy()
        
        # 1. Additive Gaussian noise with random SNR
        snr_db = np.random.choice([-5, 0, 5, 10])
        noise = np.random.randn(*patch.shape)
        signal_power = np.mean(patch**2)
        noise_power = signal_power / (10**(snr_db/10))
        patch = patch + np.sqrt(noise_power) * noise
        
        # 2. Volume scaling with random gain
        gain = np.random.uniform(0.5, 1.5)
        patch = patch * gain
        
        # 3. Compression simulation (energy smoothing)
        compression_strength = np.random.uniform(0.3, 0.5)
        smoothed = gaussian_filter(patch, sigma=0.5)
        patch = (1 - compression_strength) * patch + compression_strength * smoothed
        
        return patch
    
    def __len__(self):
        return len(self.spec_files)
    
    def __getitem__(self, idx):
        """
        Generate triplet: (anchor, positive, negative)
        With Hard Negative Mining (50% Intra-song, 50% Inter-song)
        """
        # 1. Get Anchor
        anchor = self._extract_random_patch(idx)
        
        # 2. Get Positive (Augmented version of Anchor)
        if self.transform:
            positive = self._augment_patch(anchor)
        else:
            positive = anchor.copy()
        
        # 3. Get Negative (THE FIX)
        # Flip a coin:
        # Heads (< 0.5): Pick a different song (Inter-song) -> Distinguishes Artist A vs Artist B
        # Tails (> 0.5): Pick SAME song, different time (Intra-song) -> Distinguishes Intro vs Chorus
        if random.random() < 0.5:
            # Inter-song negative (Traditional)
            negative_song_idx = idx
            while negative_song_idx == idx:
                negative_song_idx = random.randint(0, len(self.spec_files) - 1)
            negative = self._extract_random_patch(negative_song_idx)
        else:
            # Intra-song negative (Hard Negative)
            # We extract a random patch from the SAME song index.
            # Since start time is random, it will almost certainly be a different part of the song.
            negative = self._extract_random_patch(idx)
        
        # Convert to torch tensors and add channel dimension
        anchor = torch.FloatTensor(anchor).unsqueeze(0)  # (1, 128, 64)
        positive = torch.FloatTensor(positive).unsqueeze(0)
        negative = torch.FloatTensor(negative).unsqueeze(0)
        
        return anchor, positive, negative


def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4):
    """
    Create train and validation dataloaders
    """
    train_dataset = AudioFingerprintDataset(train_dir, transform=True)
    val_dataset = AudioFingerprintDataset(val_dir, transform=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader