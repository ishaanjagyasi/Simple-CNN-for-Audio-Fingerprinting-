import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import Counter
from scipy.ndimage import gaussian_filter
import random
import json
import time

from model import AudioFingerprintCNN
from database import FingerprintDatabase

class QueryMatcher:
    """Match queries against fingerprint database"""
    
    def __init__(self, database, model, device='cuda'):
        self.database = database
        self.model = model
        self.device = device
        self.model.eval()
        
    def match_query(self, query_spec, patch_size=64, batch_size=32):
        """Match query spectrogram against database"""
        n_freq, n_time = query_spec.shape
        
        if n_time < patch_size:
            return []
        
        matches = []
        
        # Process in small batches
        for start_t in range(0, n_time - patch_size + 1, batch_size):
            end_t = min(start_t + batch_size, n_time - patch_size + 1)
            
            patches = []
            query_times = []
            for t in range(start_t, end_t):
                patch = query_spec[:, t:t+patch_size]
                patches.append(patch)
                query_times.append(t)
            
            if len(patches) == 0:
                continue
            
            patches_tensor = torch.FloatTensor(np.array(patches)).unsqueeze(1).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(patches_tensor, binarize=True)
                fingerprints = (embeddings.cpu().numpy() > 0).astype(int)
            
            # Free memory
            del patches_tensor, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Query database
            for i, fp in enumerate(fingerprints):
                results = self.database.query(fp)
                for song_id, db_time in results:
                    matches.append((song_id, query_times[i], db_time))
            
            del patches, fingerprints
        
        if len(matches) == 0:
            return []
        
        # Temporal verification
        song_offsets = {}
        for song_id, query_time, db_time in matches:
            offset = query_time - db_time
            if song_id not in song_offsets:
                song_offsets[song_id] = []
            song_offsets[song_id].append(offset)
        
        song_scores = {}
        for song_id, offsets in song_offsets.items():
            offset_counts = Counter(offsets)
            best_score = max(offset_counts.values())
            song_scores[song_id] = best_score
        
        sorted_matches = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches[:1]


def apply_degradations(spec, snr_db=None, compression=None):
    """Apply degradations to spectrogram"""
    spec = spec.copy()
    
    if snr_db is not None:
        noise = np.random.randn(*spec.shape)
        signal_power = np.mean(spec**2)
        noise_power = signal_power / (10**(snr_db/10))
        spec = spec + np.sqrt(noise_power) * noise
    
    if compression is not None and compression > 0:
        spec = (1 - compression) * spec + compression * gaussian_filter(spec, sigma=0.5)
    
    return spec


def extract_query_segment(spec, duration_frames):
    """Extract random segment from spectrogram"""
    n_freq, n_time = spec.shape
    
    if n_time <= duration_frames:
        return spec
    
    max_start = n_time - duration_frames
    start_frame = np.random.randint(0, max_start + 1)
    return spec[:, start_frame:start_frame+duration_frames]


def evaluate():
    parser = argparse.ArgumentParser(description='Simplified audio fingerprinting evaluation')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--database_path', type=str, default='./fingerprint_database.pkl')
    parser.add_argument('--data_dir', type=str, default='./preprocessed/test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_songs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Load model
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = AudioFingerprintCNN().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load database
    print("Loading database...")
    database = FingerprintDatabase()
    database.load(args.database_path)
    
    # Get all test files and randomly sample
    all_files = sorted([f for f in Path(args.data_dir).glob('*_spec.npy') 
                       if not f.name.startswith('.')])
    
    if len(all_files) < args.num_songs:
        print(f"Warning: Only {len(all_files)} files available, using all")
        test_files = all_files
    else:
        test_files = random.sample(all_files, args.num_songs)
    
    print(f"Testing on {len(test_files)} randomly selected songs")
    
    # Configuration
    query_durations_sec = [5, 10, 15]
    degradations = {
        'noise_5db': {'snr_db': 5, 'compression': None},
        'noise_0db': {'snr_db': 0, 'compression': None},
        'compression_heavy': {'snr_db': None, 'compression': 0.5}
    }
    
    frames_per_second = 5000 / 58
    matcher = QueryMatcher(database, model, device)
    
    results = {}
    
    for duration_sec in query_durations_sec:
        duration_frames = int(duration_sec * frames_per_second)
        
        for deg_name, deg_params in degradations.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {duration_sec}s queries, {deg_name}")
            print(f"{'='*60}")
            
            correct = 0
            total = 0
            errors = 0
            query_times = []
            
            for spec_file in tqdm(test_files, desc=f"{deg_name}"):
                song_id = spec_file.stem.replace('_spec', '')
                
                try:
                    spec = np.load(spec_file)
                    
                    if spec.shape[1] < duration_frames:
                        continue
                    
                    # Extract random segment
                    query_spec = extract_query_segment(spec, duration_frames)
                    
                    # Apply degradations
                    query_spec = apply_degradations(query_spec, **deg_params)
                    
                    # Match query and measure time
                    start_time = time.time()
                    matches = matcher.match_query(query_spec)
                    query_time = time.time() - start_time
                    query_times.append(query_time)
                    
                    if len(matches) > 0:
                        predicted_id = matches[0][0]
                        if predicted_id == song_id:
                            correct += 1
                    
                    total += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"\nError processing {song_id}: {e}")
                    continue
            
            accuracy = (correct / total * 100) if total > 0 else 0
            avg_query_time = np.mean(query_times) if query_times else 0
            min_query_time = np.min(query_times) if query_times else 0
            max_query_time = np.max(query_times) if query_times else 0
            
            results[f"{duration_sec}s_{deg_name}"] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'errors': errors,
                'avg_query_time_ms': avg_query_time * 1000,
                'min_query_time_ms': min_query_time * 1000,
                'max_query_time_ms': max_query_time * 1000
            }
            
            print(f"\nResults: {correct}/{total} correct ({accuracy:.2f}%)")
            print(f"Query time: avg={avg_query_time*1000:.1f}ms, min={min_query_time*1000:.1f}ms, max={max_query_time*1000:.1f}ms")
            if errors > 0:
                print(f"Errors: {errors} files skipped")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for key, val in results.items():
        print(f"\n{key}:")
        print(f"  Accuracy: {val['accuracy']:6.2f}% ({val['correct']}/{val['total']})")
        print(f"  Query Time: avg={val['avg_query_time_ms']:.1f}ms, min={val['min_query_time_ms']:.1f}ms, max={val['max_query_time_ms']:.1f}ms")
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    evaluate()
