import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import pickle
from collections import defaultdict

from model import AudioFingerprintCNN

class FingerprintDatabase:
    """
    Database for storing and retrieving audio fingerprints
    Uses hash table for fast lookup
    """
    
    def __init__(self):
        """Initialize empty database"""
        self.hash_table = defaultdict(list)  # Maps fingerprint -> [(song_id, time_offset)]
        self.song_ids = []  # List of song IDs in database
        
    def add_fingerprint(self, fingerprint, song_id, time_offset):
        """
        Add a fingerprint to the database
        
        Args:
            fingerprint: 32-bit binary code (as integer or binary string)
            song_id: Song identifier
            time_offset: Time position in song (in frames)
        """
        # Convert fingerprint to integer for hashing
        if isinstance(fingerprint, np.ndarray):
            fingerprint = int(''.join(fingerprint.astype(int).astype(str)), 2)
        elif isinstance(fingerprint, str):
            fingerprint = int(fingerprint, 2)
            
        self.hash_table[fingerprint].append((song_id, time_offset))
        
    def query(self, fingerprint):
        """
        Query database for matches
        
        Args:
            fingerprint: 32-bit binary code
            
        Returns:
            List of (song_id, time_offset) tuples
        """
        if isinstance(fingerprint, np.ndarray):
            fingerprint = int(''.join(fingerprint.astype(int).astype(str)), 2)
        elif isinstance(fingerprint, str):
            fingerprint = int(fingerprint, 2)
            
        return self.hash_table.get(fingerprint, [])
    
    def save(self, filepath):
        """Save database to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'hash_table': dict(self.hash_table),
                'song_ids': self.song_ids
            }, f)
        print(f"Database saved to {filepath}")
        
    def load(self, filepath):
        """Load database from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.hash_table = defaultdict(list, data['hash_table'])
            self.song_ids = data['song_ids']
        print(f"Database loaded from {filepath}")
        
    def stats(self):
        """Print database statistics"""
        total_fingerprints = sum(len(v) for v in self.hash_table.values())
        print(f"Database Statistics:")
        print(f"  Unique fingerprints: {len(self.hash_table):,}")
        print(f"  Total fingerprints: {total_fingerprints:,}")
        print(f"  Songs: {len(self.song_ids)}")
        print(f"  Avg fingerprints per song: {total_fingerprints / len(self.song_ids):.1f}")


def build_database(model, data_dir, database_path, device='cuda', patch_size=64, max_songs=None):
    """
    Build fingerprint database from spectrograms
    
    Args:
        model: Trained CNN model
        data_dir: Directory containing test spectrograms
        database_path: Path to save database
        device: Device to run model on
        patch_size: Patch size in time frames
        
    Returns:
        FingerprintDatabase object
    """
    model.eval()
    database = FingerprintDatabase()
    
    # Get all spectrogram files
    spec_files = sorted([f for f in Path(data_dir).glob('*_spec.npy') 
                        if not f.name.startswith('.')])
    
    # Limit number of songs if specified
    if max_songs is not None:
        spec_files = spec_files[:max_songs]
    
    print(f"Building database from {len(spec_files)} songs...")
    
    with torch.no_grad():
        for spec_file in tqdm(spec_files, desc="Processing songs"):
            # Extract song ID from filename
            song_id = spec_file.stem.replace('_spec', '')
            database.song_ids.append(song_id)
            
            # Load spectrogram
            spec = np.load(spec_file)
            n_freq, n_time = spec.shape
            
            # Extract all patches with stride of 1
            patches = []
            time_offsets = []
            for t in range(0, n_time - patch_size + 1):
                patch = spec[:, t:t+patch_size]
                patches.append(patch)
                time_offsets.append(t)
            
            if len(patches) == 0:
                continue
                
            # Convert to tensor (batch, 1, 128, 64)
            patches_tensor = torch.FloatTensor(np.array(patches)).unsqueeze(1).to(device)
            
            # Generate fingerprints in batches to avoid memory issues
            batch_size = 128
            for i in range(0, len(patches_tensor), batch_size):
                batch = patches_tensor[i:i+batch_size]
                
                # Generate binary fingerprints
                embeddings = model(batch, binarize=True)
                
                # Convert to binary strings and add to database
                fingerprints = (embeddings.cpu().numpy() > 0).astype(int)
                
                for j, fp in enumerate(fingerprints):
                    time_offset = time_offsets[i + j]
                    database.add_fingerprint(fp, song_id, time_offset)
    
    # Save database
    database.save(database_path)
    database.stats()
    
    return database


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build fingerprint database')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='./preprocessed/test',
                       help='Directory containing test spectrograms')
    parser.add_argument('--database_path', type=str, default='./fingerprint_database.pkl',
                       help='Path to save database')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max_songs', type=int, default=None,
                       help='Maximum number of songs to include in database')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = AudioFingerprintCNN().to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {args.model_path}")
    
    # Build database
    build_database(model, args.data_dir, args.database_path, device, max_songs=args.max_songs)
