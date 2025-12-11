import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
DATA_DIR = Path('./fma_small')
OUTPUT_DIR = Path('./preprocessed')
SAMPLE_RATE = 5000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 58
FMIN = 300
FMAX = 2000
PATCH_TIME_FRAMES = 64
PATCH_STRIDE = 1

def preprocess_audio(audio_path, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, 
                     hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX):
    """
    Load audio and convert to log-Mel spectrogram
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate (5000 Hz)
        n_mels: Number of mel bins (128)
        n_fft: FFT window size (2048)
        hop_length: Hop length between frames (58)
        fmin: Minimum frequency (300 Hz)
        fmax: Maximum frequency (2000 Hz)
    
    Returns:
        log_mel_spec: Log-Mel spectrogram (128 x T)
    """
    # Load and resample to 5kHz
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # Apply preemphasis
    y = librosa.effects.preemphasis(y)
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, 
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def extract_patches(spectrogram, patch_time_frames=PATCH_TIME_FRAMES, stride=PATCH_STRIDE):
    """
    Extract overlapping patches from spectrogram
    
    Args:
        spectrogram: Input spectrogram (n_freq x n_time)
        patch_time_frames: Number of time frames per patch (64)
        stride: Stride between patches (1)
    
    Returns:
        patches: Array of patches (N x 128 x 64)
    """
    n_freq, n_time = spectrogram.shape
    patches = []
    
    for t in range(0, n_time - patch_time_frames + 1, stride):
        patch = spectrogram[:, t:t+patch_time_frames]
        patches.append(patch)
    
    return np.array(patches)

def process_and_save(track_id, data_dir, output_dir):
    """
    Process single track and save spectrogram
    
    Args:
        track_id: Track ID number
        data_dir: Directory containing audio files
        output_dir: Directory to save preprocessed data
    
    Returns:
        success: Boolean indicating success
    """
    # Find audio file (handle different folder structures)
    audio_path = None
    for ext in ['mp3', 'wav']:
        for pattern in [f"{track_id:06d}.{ext}", f"*/{track_id:06d}.{ext}"]:
            matches = list(data_dir.glob(pattern))
            if matches:
                audio_path = matches[0]
                break
        if audio_path:
            break
    
    if not audio_path or not audio_path.exists():
        print(f"Audio file not found for track {track_id}")
        return False
    
    try:
        spec = preprocess_audio(str(audio_path))
        np.save(output_dir / f"{track_id:06d}_spec.npy", spec)
        return True
    except Exception as e:
        print(f"Error processing {track_id}: {e}")
        return False

def main():
    """Main preprocessing pipeline"""
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("Step 1: Finding audio files...")
    # Get all audio file paths (filter out macOS hidden files)
    audio_files = []
    for ext in ['*.mp3', '*.wav']:
        audio_files.extend([f for f in DATA_DIR.glob(f'*/{ext}') if not f.name.startswith('._')])
        audio_files.extend([f for f in DATA_DIR.glob(ext) if not f.name.startswith('._')])
    
    track_ids = [int(f.stem) for f in audio_files]
    print(f"Found {len(track_ids)} audio files")
    
    # Split: 80% train, 10% val, 10% test
    print("\nStep 2: Splitting dataset...")
    train_ids, temp_ids = train_test_split(track_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    # Save splits
    np.save(OUTPUT_DIR / 'train_ids.npy', train_ids)
    np.save(OUTPUT_DIR / 'val_ids.npy', val_ids)
    np.save(OUTPUT_DIR / 'test_ids.npy', test_ids)
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Process all tracks
    print("\nStep 3: Processing audio files...")
    for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(exist_ok=True)
        
        print(f"\nProcessing {split_name} set...")
        successful = 0
        for track_id in tqdm(ids):
            if process_and_save(track_id, DATA_DIR, split_dir):
                successful += 1
        
        print(f"Successfully processed {successful}/{len(ids)} tracks")
    
    print("\nPreprocessing complete!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
