#!/usr/bin/env python3
"""
Unified Feature Extraction Script for CCP Competition

Extracts:
1. Prithvi-TL embeddings from Sentinel-2 imagery (with NaN fix)
2. VIIRS nighttime light features
3. Handcrafted Sentinel-2 spectral indices (NDVI, NDBI, NDWI, etc.)

Usage:
    python extract_features.py --split train --data_root ./data --models_root ./models
    python extract_features.py --split eval --data_root ./data --models_root ./models
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.warp import transform
from tqdm import tqdm
import warnings

# Suppress pydantic warnings from Prithvi model
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# =================================================================================================
# CONFIGURATION
# =================================================================================================
PRITHVI_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]
BAND_INDICES = [1, 2, 3, 8, 10, 11]  # B2, B3, B4, B8A, B11, B12 (0-indexed)

# =================================================================================================
# HELPER FUNCTIONS FOR SPECTRAL INDICES
# =================================================================================================
def compute_spectral_indices(bands_dict):
    """
    Compute spectral indices from Sentinel-2 bands.
    Handles NaN values in input bands.
    
    Args:
        bands_dict: dict with keys 'B3', 'B4', 'B8A', 'B11' as np arrays
    
    Returns:
        dict of spectral indices
    """
    indices = {}
    
    # Suppress divide-by-zero and invalid warnings (NaN values are expected in satellite imagery)
    with np.errstate(divide='ignore', invalid='ignore'):
        # NDVI: (NIR - Red) / (NIR + Red)
        nir = bands_dict['B8A']
        red = bands_dict['B4']
        denom = nir + red
        indices['ndvi'] = np.where(denom != 0, (nir - red) / denom, 0)
        
        # NDBI: (SWIR - NIR) / (SWIR + NIR)
        swir = bands_dict['B11']
        denom = swir + nir
        indices['ndbi'] = np.where(denom != 0, (swir - nir) / denom, 0)
        
        # NDWI: (Green - NIR) / (Green + NIR)
        green = bands_dict['B3']
        denom = green + nir
        indices['ndwi'] = np.where(denom != 0, (green - nir) / denom, 0)
    
    return indices

def compute_band_statistics(band_data, prefix=''):
    """Compute statistics for a single band/index, handling NaN values."""
    valid_data = band_data[~np.isnan(band_data)]
    
    if len(valid_data) == 0:
        return {
            f'{prefix}_mean': 0.0,
            f'{prefix}_std': 0.0,
            f'{prefix}_min': 0.0,
            f'{prefix}_max': 0.0,
            f'{prefix}_p25': 0.0,
            f'{prefix}_p50': 0.0,
            f'{prefix}_p75': 0.0,
            f'{prefix}_p90': 0.0,
        }
    
    return {
        f'{prefix}_mean': float(np.mean(valid_data)),
        f'{prefix}_std': float(np.std(valid_data)),
        f'{prefix}_min': float(np.min(valid_data)),
        f'{prefix}_max': float(np.max(valid_data)),
        f'{prefix}_p25': float(np.percentile(valid_data, 25)),
        f'{prefix}_p50': float(np.percentile(valid_data, 50)),
        f'{prefix}_p75': float(np.percentile(valid_data, 75)),
        f'{prefix}_p90': float(np.percentile(valid_data, 90)),
    }

# =================================================================================================
# CUSTOM COLLATE FUNCTION
# =================================================================================================
def custom_collate_fn(batch):
    """
    Custom collate function to properly batch dictionary fields.
    PyTorch's default collate doesn't handle dict fields well.
    """
    collated = {
        'data_id': [item['data_id'] for item in batch],
        'prithvi_input': torch.stack([item['prithvi_input'] for item in batch]),
        'lat': torch.tensor([item['lat'] for item in batch]),
        'lon': torch.tensor([item['lon'] for item in batch]),
        'year': torch.tensor([item['year'] for item in batch]),
        'doy': torch.tensor([item['doy'] for item in batch]),
        'handcrafted_features': [item['handcrafted_features'] for item in batch],
        'viirs_features': [item['viirs_features'] for item in batch],
    }
    return collated

# =================================================================================================
# DATASET CLASS
# =================================================================================================
class MultiModalDataset(Dataset):
    """Dataset that loads Sentinel-2 and VIIRS imagery for feature extraction."""
    
    def __init__(self, csv_path, composite_dir, mean, std, extract_handcrafted=True):
        self.df = pd.read_csv(csv_path)
        self.composite_dir = composite_dir
        self.mean = np.array(mean).reshape((6, 1, 1))
        self.std = np.array(std).reshape((6, 1, 1))
        self.extract_handcrafted = extract_handcrafted
        
        # Build paths
        self.df['sentinel2_path'] = self.df['sentinel2_tiff_file_name'].apply(
            lambda x: os.path.join(self.composite_dir, x)
        )
        self.df['viirs_path'] = self.df['viirs_tiff_file_name'].apply(
            lambda x: os.path.join(self.composite_dir, x)
        )

    def __len__(self):
        return len(self.df)

    def _get_time_info(self, quarter_str):
        """Convert 'YYYY-Q#' to year and day-of-year."""
        year, quarter = quarter_str.split('-')
        year = int(year)
        q_num = int(quarter[1])
        doy_map = {1: 45, 2: 135, 3: 225, 4: 315}
        doy = doy_map.get(q_num, 1)
        return year, doy

    def _get_lat_lon(self, src):
        """Extract center lat/lon from raster. Since CRS is already EPSG:4326, no transform needed."""
        bounds = src.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.top + bounds.bottom) / 2
        # Data is already in EPSG:4326, so no transform needed (fixes Bug 2)
        return center_y, center_x  # lat, lon

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentinel2_path = row['sentinel2_path']
        viirs_path = row['viirs_path']
        quarter_str = row['quarter_label']
        
        # Initialize outputs
        prithvi_input = None
        handcrafted_features = {}
        viirs_features = {}
        lat, lon = 0.0, 0.0
        year, doy = 2000, 1
        
        try:
            # ===== SENTINEL-2 PROCESSING =====
            with rasterio.open(sentinel2_path) as src:
                # Read Prithvi bands
                bands_to_read = [i+1 for i in BAND_INDICES]  # 1-indexed for rasterio
                img = src.read(bands_to_read).astype(np.float32)  # (6, H, W)
                
                # Get metadata
                lat, lon = self._get_lat_lon(src)
                year, doy = self._get_time_info(quarter_str)
                
                # Normalize
                img_normalized = (img - self.mean) / self.std
                
                # FIX BUG 1: Replace NaN with 0.0 (= band mean in raw space)
                img_normalized = np.nan_to_num(img_normalized, nan=0.0)
                
                # Prepare for Prithvi
                img_tensor = torch.from_numpy(img_normalized.astype(np.float32))
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                ).squeeze(0)
                img_tensor = img_tensor.unsqueeze(1)  # (6, 1, 224, 224)
                prithvi_input = img_tensor
                
                # ===== HANDCRAFTED FEATURES =====
                if self.extract_handcrafted:
                    # Read all 12 bands for spectral indices
                    all_bands = src.read().astype(np.float32)  # (12, H, W)
                    
                    # Extract specific bands (1-indexed in naming convention)
                    bands_dict = {
                        'B2': all_bands[1],   # Blue
                        'B3': all_bands[2],   # Green
                        'B4': all_bands[3],   # Red
                        'B8A': all_bands[8],  # NIR
                        'B11': all_bands[10], # SWIR1
                        'B12': all_bands[11], # SWIR2
                    }
                    
                    # Compute spectral indices
                    indices = compute_spectral_indices(bands_dict)
                    
                    # Statistics for each index
                    for idx_name, idx_data in indices.items():
                        handcrafted_features.update(compute_band_statistics(idx_data, prefix=idx_name))
                    
                    # Statistics for raw bands
                    band_names = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
                    for band_name in band_names:
                        handcrafted_features.update(
                            compute_band_statistics(bands_dict[band_name], prefix=f'band_{band_name}')
                        )
                    
                    # Valid pixel ratio (inverse of cloud/NaN coverage)
                    total_pixels = all_bands[0].size
                    valid_pixels = np.sum(~np.isnan(all_bands[0]))
                    handcrafted_features['valid_pixel_ratio'] = valid_pixels / total_pixels
            
            # ===== VIIRS PROCESSING =====
            with rasterio.open(viirs_path) as src:
                viirs_data = src.read(1).astype(np.float32)  # (H, W)
                
                # Handle NaN
                viirs_stats = compute_band_statistics(viirs_data, prefix='viirs')
                viirs_features.update(viirs_stats)
                
                # Additional VIIRS-specific features
                valid_data = viirs_data[~np.isnan(viirs_data)]
                if len(valid_data) > 0:
                    # Proportion of lit pixels (radiance > 1.0)
                    viirs_features['viirs_lit_ratio'] = float(np.sum(valid_data > 1.0) / len(valid_data))
                    # High radiance ratio (> 10.0)
                    viirs_features['viirs_high_rad_ratio'] = float(np.sum(valid_data > 10.0) / len(valid_data))
                else:
                    viirs_features['viirs_lit_ratio'] = 0.0
                    viirs_features['viirs_high_rad_ratio'] = 0.0
                    
        except Exception as e:
            print(f"Error processing sample {row['data_id']}: {e}")
            # Return zero/dummy values
            prithvi_input = torch.zeros((6, 1, 224, 224))
        
        return {
            'data_id': row['data_id'],
            'prithvi_input': prithvi_input,
            'lat': lat,
            'lon': lon,
            'year': year,
            'doy': doy,
            'handcrafted_features': handcrafted_features,
            'viirs_features': viirs_features,
        }

# =================================================================================================
# MODEL LOADING
# =================================================================================================
def load_prithvi_model(checkpoint_path, device):
    """Load Prithvi-MAE model for feature extraction."""
    try:
        from prithvi_mae import PrithviMAE
    except ImportError:
        raise ImportError("Could not import PrithviMAE. Check your model path.")
    
    config_args = {
        "img_size": 224, "num_frames": 1, "patch_size": [1, 16, 16], "in_chans": 6,
        "embed_dim": 1024, "depth": 24, "num_heads": 16, "decoder_embed_dim": 512,
        "decoder_depth": 8, "decoder_num_heads": 16, "mlp_ratio": 4,
        "coords_encoding": ["time", "location"], "coords_scale_learn": True,
        "mask_ratio": 0.75, "norm_pix_loss": False
    }
    
    model = PrithviMAE(**config_args)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Remove position embedding keys due to T mismatch (trained T=4, inference T=1)
        keys_to_ignore = [k for k in state_dict.keys() if 'pos_embed' in k]
        if keys_to_ignore:
            print(f"Ignoring {len(keys_to_ignore)} pos_embed keys due to shape mismatch")
            for k in keys_to_ignore:
                del state_dict[k]
        
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found.")
    
    model.to(device)
    model.eval()
    return model

# =================================================================================================
# MAIN EXTRACTION
# =================================================================================================
def extract_features(args):
    """Main feature extraction pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup paths
    data_root = args.data_root
    
    if args.split == 'train':
        csv_path = os.path.join(data_root, 'train_dataset', 'train_tabular.csv')
        composite_dir = os.path.join(data_root, 'train_dataset', 'train_composite')
        emb_out = 'prithvi_embeddings_train.npy'
        ids_out = 'prithvi_embeddings_ids_train.npy'
        handcrafted_out = 'handcrafted_features_train.csv'
    else:  # eval
        csv_path = os.path.join(data_root, 'evaluation_dataset', 'evaluation_tabular_no_target.csv')
        composite_dir = os.path.join(data_root, 'evaluation_dataset', 'evaluation_composite')
        emb_out = 'prithvi_embeddings_eval.npy'
        ids_out = 'prithvi_embeddings_ids_eval.npy'
        handcrafted_out = 'handcrafted_features_eval.csv'
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Dataset
    dataset = MultiModalDataset(csv_path, composite_dir, PRITHVI_MEAN, PRITHVI_STD,
                                  extract_handcrafted=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=custom_collate_fn)
    
    # Load Prithvi model
    model_ckpt = os.path.join(args.models_root, 'Prithvi-EO-2.0-300M-TL/Prithvi_EO_V2_300M_TL.pt')
    sys.path.append(os.path.join(args.models_root, 'Prithvi-EO-2.0-300M-TL'))
    model = load_prithvi_model(model_ckpt, device)
    
    # Storage
    embeddings = []
    data_ids = []
    handcrafted_rows = []
    
    print(f"Extracting features for {args.split} split...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Prithvi embeddings
            pixel_values = batch['prithvi_input'].float().to(device)
            B = pixel_values.shape[0]
            T = 1
            
            lat = batch['lat'].float().to(device).view(B, 1)
            lon = batch['lon'].float().to(device).view(B, 1)
            location_coords = torch.cat([lat, lon], dim=1)
            
            year = batch['year'].float().to(device).view(B, 1, 1).expand(B, T, 1)
            doy = batch['doy'].float().to(device).view(B, 1, 1).expand(B, T, 1)
            temporal_coords = torch.cat([year, doy], dim=2)
            
            features = model.forward_features(pixel_values, temporal_coords, location_coords)
            last_features = features[-1]
            emb = torch.mean(last_features[:, 1:, :], dim=1)  # Mean pooling over patches
            
            embeddings.append(emb.cpu().numpy())
            data_ids.extend(batch['data_id'])
            
            # Handcrafted features
            for i in range(B):
                row = {'data_id': batch['data_id'][i]}
                row['lat'] = float(batch['lat'][i])
                row['lon'] = float(batch['lon'][i])
                row.update(batch['handcrafted_features'][i])
                row.update(batch['viirs_features'][i])
                handcrafted_rows.append(row)
    
    # Save embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    print(f"NaN check: {np.isnan(embeddings).sum()} NaN values (should be 0)")
    
    np.save(emb_out, embeddings)
    np.save(ids_out, np.array(data_ids))
    print(f"Saved embeddings to {emb_out}")
    
    # Save handcrafted features
    handcrafted_df = pd.DataFrame(handcrafted_rows)
    handcrafted_df.to_csv(handcrafted_out, index=False)
    print(f"Saved handcrafted features to {handcrafted_out}")

# =================================================================================================
# CLI
# =================================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified feature extraction for CCP competition')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'eval'],
                        help='Data split to process')
    parser.add_argument('--data_root', type=str, default='/content/drive/MyDrive/Competitions/Solafune/CCP/data',
                        help='Data root directory')
    parser.add_argument('--models_root', type=str, default='/content/drive/MyDrive/Competitions/Solafune/CCP/models',
                        help='Models root directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    
    args = parser.parse_args()
    extract_features(args)
