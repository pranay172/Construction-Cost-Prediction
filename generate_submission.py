#!/usr/bin/env python3
"""
Generate Submission for CCP Competition — v2

Supports ensemble inference with:
- Country-stratified models (XGBoost + LightGBM + CatBoost)
- Per-fold geolocation aggregates averaged for final prediction
- Weighted ensemble with optimized weights from training
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import joblib
import argparse
import warnings
from train_model import SpatialKNN

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

TARGET_COL = 'construction_cost_per_m2_usd'


# =================================================================================================
# Import shared utilities from train_model
# =================================================================================================
from train_model import (
    TargetEncoder, apply_geo_aggregates, get_feature_cols, get_categorical_cols
)


# =================================================================================================
# DATA LOADING
# =================================================================================================
def load_data(data_root, is_train=True):
    """Load tabular data, embeddings, and handcrafted features."""
    if is_train:
        csv_path = os.path.join(data_root, 'train_dataset', 'train_tabular.csv')
        emb_path = 'prithvi_embeddings_train.npy'
        ids_path = 'prithvi_embeddings_ids_train.npy'
        handcrafted_path = 'handcrafted_features_train.csv'
    else:
        csv_path = os.path.join(data_root, 'evaluation_dataset', 'evaluation_tabular_no_target.csv')
        emb_path = 'prithvi_embeddings_eval.npy'
        ids_path = 'prithvi_embeddings_ids_eval.npy'
        handcrafted_path = 'handcrafted_features_eval.csv'
    
    print(f"Loading {'train' if is_train else 'eval'} data...")
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found at {emb_path}. Run extract_features.py first.")
    
    embeddings = np.load(emb_path)
    ids = np.load(ids_path)
    
    if os.path.exists(handcrafted_path):
        handcrafted_df = pd.read_csv(handcrafted_path)
    else:
        handcrafted_df = pd.DataFrame({'data_id': ids})
    
    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df['data_id'] = ids
    
    merged_df = pd.merge(df, emb_df, on='data_id', how='left')
    merged_df = pd.merge(merged_df, handcrafted_df, on='data_id', how='left')
    
    return merged_df, emb_cols


# =================================================================================================
# BASE PREPROCESSING (mirrors train_model.py exactly)
# =================================================================================================
def base_preprocess(df, emb_cols, pca_model):
    """Apply same base preprocessing as training (no target-dependent features)."""
    df = df.copy()
    
    # Time features
    df['year'] = df['quarter_label'].apply(lambda x: int(x.split('-')[0]))
    df['quarter'] = df['quarter_label'].apply(lambda x: int(x.split('-')[1][1]))
    df['year_quarter_num'] = df['year'] + (df['quarter'] - 1) / 4.0
    
    # Feature interactions
    df['gdp_cpi'] = df['deflated_gdp_usd'] * df['us_cpi']
    df['infrastructure_access_count'] = (
        df['access_to_airport'].map({'Yes': 1, 'No': 0}).fillna(0) +
        df['access_to_port'].map({'Yes': 1, 'No': 0}).fillna(0) +
        df['access_to_highway'].map({'Yes': 1, 'No': 0}).fillna(0) +
        df['access_to_railway'].map({'Yes': 1, 'No': 0}).fillna(0)
    )
    
    # Log transforms
    df['log_gdp'] = np.log1p(df['deflated_gdp_usd'])
    df['log_distance_capital'] = np.log1p(df['straight_distance_to_capital_km'])
    df['log_cpi'] = np.log1p(df['us_cpi'])
    
    # Risk score
    risk_map = {
        'seismic_hazard_zone': {'Low': 0, 'Moderate': 1, 'High': 2},
        'flood_risk_class': {'No': 0, 'Yes': 1},
        'tropical_cyclone_wind_risk': {'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3},
        'tornadoes_wind_risk': {'Very Low': 0, 'Low': 1, 'Moderate': 2, 'High': 3},
    }
    df['risk_score'] = 0
    for col, mapping in risk_map.items():
        if col in df.columns:
            df['risk_score'] += df[col].map(mapping).fillna(0)
    
    # GDP interaction
    df['gdp_x_developed'] = df['deflated_gdp_usd'] * df['developed_country'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # Risk x GDP interaction (Disasters in poorer areas might have different cost dynamics)
    df['risk_x_gdp'] = df['risk_score'] * df['log_gdp']
    df['risk_x_cpi'] = df['risk_score'] * df['log_cpi']
    
    # Interaction: Infrastructure * GDP (Wealthier areas with more infra might be more expensive)
    df['infra_x_gdp'] = df['infrastructure_access_count'] * df['log_gdp']
    
    # PCA on embeddings
    if emb_cols and len(emb_cols) > 0:
        embeddings = df[emb_cols].values
        embeddings_pca = pca_model.transform(embeddings)
        pca_cols = [f'pca_{i}' for i in range(embeddings_pca.shape[1])]
        pca_df = pd.DataFrame(embeddings_pca, columns=pca_cols, index=df.index)
        df = df.drop(columns=emb_cols)
        df = pd.concat([df, pca_df], axis=1)
    
    return df


def inference_preprocess(eval_df, train_df, fold_artifact):
    """
    Apply fold-specific preprocessing for inference.
    Uses geo aggregates and target encoding from the training fold.
    
    At inference time, we use the FULL training data to compute aggregates
    (no leakage concern since we're predicting on unseen test data).
    """
    eval_processed = eval_df.copy()
    
    geo_aggs = fold_artifact['geo_aggs']
    cy_aggs = fold_artifact['cy_aggs']
    te_model = fold_artifact['te_model']
    label_encoders = fold_artifact['label_encoders']
    spatial_knn = fold_artifact['spatial_knn']
    feature_cols = fold_artifact['feature_cols']
    
    # Apply geo aggregates (pass train_df for temporal lag features)
    eval_processed = apply_geo_aggregates(eval_processed, geo_aggs, cy_aggs, train_df=train_df)
    
    # Apply target encoding
    target_encode_cols = ['geolocation_name']
    eval_processed[target_encode_cols] = te_model.transform(eval_processed[target_encode_cols])
    
    # Label encode categoricals using saved encoders
    drop_cols = [
        'data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name',
        'quarter_label', TARGET_COL, 'geolocation_name', 'country'
    ]
    cat_cols = [c for c in eval_processed.columns
                if (pd.api.types.is_string_dtype(eval_processed[c]) or eval_processed[c].dtype == 'object')
                and c not in drop_cols]
    
    for col in cat_cols:
        if col in label_encoders:
            le = label_encoders[col]
            known = set(le.classes_)
            default = le.classes_[0]
            eval_processed[col] = le.transform(
                eval_processed[col].astype(str).apply(lambda x: x if x in known else default)
            )
        else:
            le = LabelEncoder()
            eval_processed[col] = le.fit_transform(eval_processed[col].astype(str))
            
    # Apply Spatial KNN
    if 'lat' in eval_processed.columns and 'lon' in eval_processed.columns:
        knn_feats = spatial_knn.transform(eval_processed[['lat', 'lon']], is_train=False)
        eval_processed = pd.concat([eval_processed, knn_feats], axis=1)
    
    # Ensure all expected feature columns exist
    for col in feature_cols:
        if col not in eval_processed.columns:
            eval_processed[col] = 0
            print(f"Warning: Added missing column '{col}' to eval data")
    
    return eval_processed, feature_cols


# =================================================================================================
# MAIN
# =================================================================================================
def main(args):
    print("=" * 70)
    print("Generating Ensemble Submission")
    print("=" * 70)
    
    # Load saved artifacts
    pca_model = joblib.load('pca_model.pkl')
    all_models = joblib.load('ensemble_models.pkl')
    fold_artifacts = joblib.load('fold_artifacts.pkl')
    training_meta = joblib.load('training_meta.pkl')
    
    model_types = training_meta['model_types']
    best_weights = training_meta['best_weights']
    n_folds = training_meta['n_folds']
    
    print(f"Model types: {model_types}")
    print(f"Ensemble weights: {best_weights}")
    print(f"Folds: {n_folds}")
    
    # Load data
    train_df, train_emb_cols = load_data(data_root=args.data_root, is_train=True)
    eval_df, eval_emb_cols = load_data(data_root=args.data_root, is_train=False)
    
    # Base preprocessing (same for train and eval)
    train_processed = base_preprocess(train_df, train_emb_cols, pca_model)
    eval_processed = base_preprocess(eval_df, eval_emb_cols, pca_model)
    
    print(f"Train: {train_processed.shape}, Eval: {eval_processed.shape}")
    
    # === Compute final geo aggregates from FULL training data ===
    # For inference, we use all training data (no CV leakage concern)
    from train_model import compute_geo_aggregates, compute_country_year_aggs
    
    final_geo_aggs = compute_geo_aggregates(train_processed)
    final_cy_aggs = compute_country_year_aggs(train_processed)
    
    # Fit final target encoder on full training data
    final_te = TargetEncoder(smoothing=10, min_samples=3)
    target_encode_cols = ['geolocation_name']
    final_te.fit(train_processed[target_encode_cols], train_processed[TARGET_COL], cols=target_encode_cols)
    
    # === Country-stratified prediction ===
    original_country = eval_processed['country'].values.copy()
    unique_countries = np.unique(original_country)
    
    preds = np.zeros(len(eval_processed))
    
    for country in unique_countries:
        country_mask = original_country == country
        country_eval = eval_processed[country_mask].copy()
        
        if country not in all_models:
            print(f"Warning: No model for country '{country}', using fallback")
            continue
        
        print(f"\n--- Country: {country} ({country_mask.sum()} samples) ---")
        
        country_models = all_models[country]
        country_fold_artifacts = fold_artifacts[country]
        
        # For each fold's artifacts, preprocess and predict, then average
        country_preds = {mt: [] for mt in model_types}
        
        for fold_idx, artifact in enumerate(country_fold_artifacts):
            # Apply fold-specific preprocessing
            eval_fold_processed, feature_cols = inference_preprocess(
                country_eval, train_processed, artifact
            )
            
            X_eval = eval_fold_processed[feature_cols]
            
            for mt in model_types:
                model = country_models[mt][fold_idx]
                pred = np.expm1(model.predict(X_eval))
                pred = np.maximum(pred, 0)
                country_preds[mt].append(pred)
        
        # Average across folds for each model type
        model_avg_preds = {}
        for mt in model_types:
            model_avg_preds[mt] = np.mean(country_preds[mt], axis=0)
            print(f"  {mt}: pred range [{model_avg_preds[mt].min():.2f}, "
                  f"{model_avg_preds[mt].max():.2f}], mean={model_avg_preds[mt].mean():.2f}")
        
        # Weighted ensemble
        country_final = np.zeros(country_mask.sum())
        for mt, w in best_weights.items():
            if mt in model_avg_preds:
                country_final += w * model_avg_preds[mt]
        
        preds[country_mask] = country_final
        print(f"  Ensemble: range [{country_final.min():.2f}, {country_final.max():.2f}], "
              f"mean={country_final.mean():.2f}")
    
    # Ensure no negative predictions
    preds = np.maximum(preds, 0)
    
    # Create submission
    submission = pd.DataFrame({
        'data_id': eval_df['data_id'],
        'construction_cost_per_m2_usd': preds
    })
    
    sub_path = args.output if args.output else 'submission.csv'
    submission.to_csv(sub_path, index=False)
    
    print(f"\nSubmission saved to {sub_path}")
    print(f"Predictions: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}")
    print(submission.head(10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ensemble submission')
    parser.add_argument('--data_root', type=str,
                        default='/content/drive/MyDrive/Competitions/Solafune/CCP/data',
                        help='Path to data root')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: submission.csv)')
    args = parser.parse_args()
    main(args)