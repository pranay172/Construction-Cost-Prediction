#!/usr/bin/env python3
"""
Enhanced Training Script for CCP Competition — v2

Key Improvements over v1:
- 3-model ensemble: XGBoost + LightGBM + CatBoost
- In-fold target encoding & geolocation aggregates (no leakage)
- Rich geolocation-level features (mean, median, std, trend, lag)
- Enhanced feature interactions (GDP growth, risk score, CPI-adjusted)
- Country-stratified training with proper OOF evaluation
- Optimizes log1p(target) with reg:squarederror for stable RMSLE
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import joblib
import argparse
import warnings
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

TARGET_COL = 'construction_cost_per_m2_usd'

# =================================================================================================
# GEOLOCATION AGGREGATE FEATURES (computed from training data only)
# =================================================================================================
def compute_geo_aggregates(train_df, geo_col='geolocation_name', target_col=TARGET_COL):
    """
    Compute rich geolocation-level aggregate features from the training fold only.
    These exploit the fact that 100% of geolocations overlap between train and eval.
    
    Returns a DataFrame indexed by geolocation with aggregate features.
    """
    groups = train_df.groupby(geo_col)
    
    aggs = pd.DataFrame()
    aggs['geo_target_mean'] = groups[target_col].mean()
    aggs['geo_target_median'] = groups[target_col].median()
    aggs['geo_target_std'] = groups[target_col].std().fillna(0)
    aggs['geo_target_min'] = groups[target_col].min()
    aggs['geo_target_max'] = groups[target_col].max()
    aggs['geo_target_count'] = groups[target_col].count()
    aggs['geo_target_range'] = aggs['geo_target_max'] - aggs['geo_target_min']
    
    # Log-space statistics (more relevant for RMSLE)
    log_target = np.log1p(train_df[target_col])
    log_groups = log_target.groupby(train_df[geo_col])
    aggs['geo_log_target_mean'] = log_groups.mean()
    aggs['geo_log_target_std'] = log_groups.std().fillna(0)
    
    # Temporal trend: linear slope of target vs time within each geolocation
    def compute_slope(group):
        if len(group) < 3:
            return 0.0
        year = group['year'].values
        quarter = group['quarter_label'].apply(lambda x: int(x.split('-')[1][1])).values
        time = year + (quarter - 1) / 4.0
        target = group[target_col].values
        if np.std(time) == 0:
            return 0.0
        slope, _, _, _, _ = scipy_stats.linregress(time, target)
        return slope
    
    slopes = train_df.groupby(geo_col).apply(compute_slope, include_groups=False)
    aggs['geo_target_slope'] = slopes
    
    # GDP-normalized cost per geolocation
    train_df_temp = train_df.copy()
    train_df_temp['cost_per_gdp'] = train_df_temp[target_col] / train_df_temp['deflated_gdp_usd']
    aggs['geo_cost_per_gdp_mean'] = train_df_temp.groupby(geo_col)['cost_per_gdp'].mean()
    
    # Percentile features
    aggs['geo_target_p25'] = groups[target_col].quantile(0.25)
    aggs['geo_target_p75'] = groups[target_col].quantile(0.75)
    aggs['geo_target_iqr'] = aggs['geo_target_p75'] - aggs['geo_target_p25']
    
    # Coefficient of variation (volatility)
    aggs['geo_target_cv'] = (aggs['geo_target_std'] / aggs['geo_target_mean'].clip(lower=1e-6))
    
    # Average risk score per geolocation (if available)
    if 'risk_score' in train_df.columns:
        aggs['geo_risk_mean'] = groups['risk_score'].mean()
        aggs['geo_risk_max'] = groups['risk_score'].max()
    
    return aggs.reset_index()


def compute_country_year_aggs(train_df, target_col=TARGET_COL):
    """Compute country × year aggregate features."""
    groups = train_df.groupby(['country', 'year'])
    
    cy_aggs = pd.DataFrame()
    cy_aggs['country_year_target_mean'] = groups[target_col].mean()
    cy_aggs['country_year_target_std'] = groups[target_col].std().fillna(0)
    cy_aggs['country_year_target_median'] = groups[target_col].median()
    cy_aggs['country_year_count'] = groups[target_col].count()
    
    return cy_aggs.reset_index()


def apply_geo_aggregates(df, geo_aggs, cy_aggs, train_df=None, geo_col='geolocation_name'):
    """Merge geolocation and country-year aggregates into the dataframe."""
    df = df.copy()
    
    # Merge geo aggregates
    df = pd.merge(df, geo_aggs, on=geo_col, how='left')
    
    # Merge country-year aggregates
    df = pd.merge(df, cy_aggs, on=['country', 'year'], how='left')
    
    # Fill NaN for unseen geolocations/years (shouldn't happen with 100% overlap)
    geo_fill_cols = [c for c in df.columns if c.startswith('geo_')]
    cy_fill_cols = [c for c in df.columns if c.startswith('country_year_')]
    
    for col in geo_fill_cols + cy_fill_cols:
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    
    # Rank of geolocation within country
    if 'geo_target_mean' in df.columns:
        df['geo_country_rank'] = df.groupby('country')['geo_target_mean'].rank(pct=True)
    
    # Deviation of this sample's geo mean from country-year mean
    if 'geo_target_mean' in df.columns and 'country_year_target_mean' in df.columns:
        df['geo_deviation_from_cy'] = df['geo_target_mean'] - df['country_year_target_mean']
        # Log-space deviation
        if 'geo_log_target_mean' in df.columns:
            df['geo_log_deviation_from_cy'] = df['geo_log_target_mean'] - np.log1p(df['country_year_target_mean'])
    
    # Geo×quarter temporal lag features (from training data)
    if train_df is not None:
        df = _add_temporal_lag_features(df, train_df, geo_col)
    
    # Expected target based on slope (geo_mean + slope * time_delta)
    if 'geo_target_slope' in df.columns and 'year_quarter_num' in df.columns:
        if train_df is not None and 'year_quarter_num' in train_df.columns:
            geo_mean_time = train_df.groupby(geo_col)['year_quarter_num'].mean()
            df['geo_mean_time'] = df[geo_col].map(geo_mean_time).fillna(df['year_quarter_num'])
            df['geo_target_expected'] = df['geo_target_mean'] + df['geo_target_slope'] * (df['year_quarter_num'] - df['geo_mean_time'])
            df.drop(columns=['geo_mean_time'], inplace=True)
    
    return df


def _add_temporal_lag_features(df, train_df, geo_col='geolocation_name'):
    """Add temporal lag features: target from nearest previous quarter for same geolocation."""
    TARGET_COL_LOCAL = 'construction_cost_per_m2_usd'
    
    if TARGET_COL_LOCAL not in train_df.columns:
        return df
    
    # Build lookup: geo → sorted list of (year_quarter_num, target)
    train_temp = train_df[[geo_col, 'year_quarter_num', TARGET_COL_LOCAL]].copy()
    train_temp = train_temp.sort_values('year_quarter_num')
    
    geo_history = {}
    for geo, group in train_temp.groupby(geo_col):
        times = group['year_quarter_num'].values
        targets = group[TARGET_COL_LOCAL].values
        geo_history[geo] = (times, targets)
    
    # For each row in df, find the most recent previous quarter's target
    lag1_values = []
    lag_diff_values = []
    
    for _, row in df.iterrows():
        geo = row[geo_col]
        t = row['year_quarter_num']
        
        if geo in geo_history:
            times, targets = geo_history[geo]
            # Find most recent previous (strictly before current time)
            mask = times < t - 0.01  # Small epsilon to avoid matching same quarter
            if mask.any():
                idx = np.where(mask)[0][-1]  # Last one before current
                lag1_values.append(targets[idx])
                lag_diff_values.append(targets[idx] - row.get('geo_target_mean', targets[idx]))
            else:
                lag1_values.append(np.nan)
                lag_diff_values.append(np.nan)
        else:
            lag1_values.append(np.nan)
            lag_diff_values.append(np.nan)
    
    df['geo_lag1_target'] = lag1_values
    df['geo_lag1_diff'] = lag_diff_values
    
    # Fill NaN with geo mean
    df['geo_lag1_target'] = df['geo_lag1_target'].fillna(df['geo_target_mean'])
    df['geo_lag1_diff'] = df['geo_lag1_diff'].fillna(0)
    
    return df


# =================================================================================================
# TARGET ENCODING (in-fold safe)
# =================================================================================================
class TargetEncoder:
    """Target encoder with smoothing. Must be fit inside CV folds only."""
    
    def __init__(self, smoothing=10, min_samples=5):
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.encodings = {}
        self.global_mean = None
    
    def fit(self, X, y, cols=None):
        self.global_mean = y.mean()
        if cols is None:
            cols = [c for c in X.columns if pd.api.types.is_string_dtype(X[c]) or X[c].dtype == 'object']
        
        for col in cols:
            if col not in X.columns:
                continue
            stats = y.groupby(X[col]).agg(['mean', 'count'])
            smoothed_mean = (
                stats['count'] * stats['mean'] + self.smoothing * self.global_mean
            ) / (stats['count'] + self.smoothing)
            smoothed_mean = np.where(
                stats['count'] < self.min_samples,
                self.global_mean,
                smoothed_mean
            )
            smoothed_mean = pd.Series(smoothed_mean, index=stats.index)
            self.encodings[col] = smoothed_mean.to_dict()
        return self
    
    def transform(self, X, cols=None):
        X_encoded = X.copy()
        for col in self.encodings:
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].map(self.encodings[col]).fillna(self.global_mean).astype(float)
        return X_encoded


# =================================================================================================
# SPATIAL KNN FEATURES
# =================================================================================================
class SpatialKNN:
    """
    Generates spatial features using KNN on Lat/Lon.
    - Fits on (Lat, Lon, Target).
    - Generates features: Mean/Distance of K nearest neighbors.
    - Handles Leave-One-Out (LOO) for training data to prevent leakage.
    """
    
    def __init__(self, n_neighbors=[5, 10, 20]):
        self.n_neighbors = n_neighbors
        self.max_k = max(n_neighbors)
        self.model = None
        self.y_train = None
    
    def fit(self, X_latlon, y):
        self.model = KNeighborsRegressor(n_neighbors=self.max_k + 1, metric='haversine')
        # Convert lat/lon to radians for haversine
        X_rad = np.radians(X_latlon)
        self.model.fit(X_rad, y)
        self.y_train = y.values
        return self
    
    def transform(self, X_latlon, is_train=False):
        """
        Args:
            X_latlon: DataFrame/Array with [lat, lon]
            is_train: If True, excludes the self-neighbor (index 0) to prevent leakage.
        """
        X_rad = np.radians(X_latlon)
        
        # Find K+1 neighbors if training (to skip self), else K
        k_query = self.max_k + 1 if is_train else self.max_k
        distances, indices = self.model.kneighbors(X_rad, n_neighbors=k_query)
        
        # Convert distances back to km (approx)
        # Earth radius ~ 6371 km
        distances = distances * 6371.0
        
        features = pd.DataFrame(index=X_latlon.index)
        
        for k in self.n_neighbors:
            # If train, we skip the first neighbor (itself) -> use indices 1..k+1
            # If test, we use indices 0..k
            start_idx = 1 if is_train else 0
            end_idx = start_idx + k
            
            # Gather targets of neighbors
            # indices shape: (N, max_k+1) or (N, max_k)
            # We need to vectorized select
            k_indices = indices[:, start_idx:end_idx]
            k_dists = distances[:, start_idx:end_idx]
            
            # Fetch targets
            # k_indices is (N, k), flat it to index y_train
            neighbor_targets = self.y_train[k_indices] # (N, k)
            
            # Features
            features[f'knn_mean_{k}'] = np.mean(neighbor_targets, axis=1)
            features[f'knn_std_{k}'] = np.std(neighbor_targets, axis=1)
            features[f'knn_dist_mean_{k}'] = np.mean(k_dists, axis=1)
        
        return features


# =================================================================================================
# DATA LOADING
# =================================================================================================
def load_data(data_root):
    """Load tabular data, embeddings, and handcrafted features."""
    print("Loading data...")
    
    train_csv = os.path.join(data_root, 'train_dataset', 'train_tabular.csv')
    df = pd.read_csv(train_csv)
    
    emb_path = 'prithvi_embeddings_train.npy'
    ids_path = 'prithvi_embeddings_ids_train.npy'
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError("Embeddings not found. Run extract_features.py first.")
    
    embeddings = np.load(emb_path)
    ids = np.load(ids_path)
    print(f"Embeddings shape: {embeddings.shape}, NaN count: {np.isnan(embeddings).sum()}")
    
    handcrafted_path = 'handcrafted_features_train.csv'
    if os.path.exists(handcrafted_path):
        handcrafted_df = pd.read_csv(handcrafted_path)
        print(f"Handcrafted features: {handcrafted_df.shape[1] - 1} columns")
    else:
        print("Warning: Handcrafted features not found")
        handcrafted_df = pd.DataFrame({'data_id': ids})
    
    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df['data_id'] = ids
    
    merged_df = pd.merge(df, emb_df, on='data_id', how='left')
    merged_df = pd.merge(merged_df, handcrafted_df, on='data_id', how='left')
    
    print(f"Final merged shape: {merged_df.shape}")
    return merged_df, emb_cols


# =================================================================================================
# BASE PREPROCESSING (features that don't leak — applied once)
# =================================================================================================
def base_preprocess(df, emb_cols, pca_components=64, seed=42,
                    fit_pca=True, pca_model=None):
    """
    Base preprocessing: time features, interactions, PCA on embeddings.
    NO target encoding or geo aggregates here — those go inside CV folds.
    """
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
    
    # Log transform skewed features
    df['log_gdp'] = np.log1p(df['deflated_gdp_usd'])
    df['log_distance_capital'] = np.log1p(df['straight_distance_to_capital_km'])
    df['log_cpi'] = np.log1p(df['us_cpi'])
    
    # Risk score: composite of hazard features
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
    
    # GDP per capita proxy interaction
    df['gdp_x_developed'] = df['deflated_gdp_usd'] * df['developed_country'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # Risk x GDP interaction (Disasters in poorer areas might have different cost dynamics)
    df['risk_x_gdp'] = df['risk_score'] * df['log_gdp']
    df['risk_x_cpi'] = df['risk_score'] * df['log_cpi']
    
    # Interaction: Infrastructure * GDP (Wealthier areas with more infra might be more expensive)
    df['infra_x_gdp'] = df['infrastructure_access_count'] * df['log_gdp']
    
    # === PCA on Embeddings ===
    if emb_cols and len(emb_cols) > 0:
        embeddings = df[emb_cols].values
        
        if fit_pca:
            pca_model = PCA(n_components=pca_components, random_state=seed)
            embeddings_pca = pca_model.fit_transform(embeddings)
            print(f"PCA: {embeddings.shape[1]} -> {embeddings_pca.shape[1]} dims, "
                  f"explained variance: {pca_model.explained_variance_ratio_.sum():.2%}")
        else:
            embeddings_pca = pca_model.transform(embeddings)
        
        pca_cols = [f'pca_{i}' for i in range(embeddings_pca.shape[1])]
        pca_df = pd.DataFrame(embeddings_pca, columns=pca_cols, index=df.index)
        df = df.drop(columns=emb_cols)
        df = pd.concat([df, pca_df], axis=1)
    
    return df, pca_model


def fold_preprocess(train_fold_df, val_fold_df, cat_cols_to_encode,
                    fit_label_encoders=True, label_encoders=None):
    """
    In-fold preprocessing: target encoding + geolocation aggregates.
    Fitted on train_fold ONLY, applied to both train and val.
    
    Returns train_df, val_df, fold artifacts (te_model, geo_aggs, cy_aggs, label_encoders)
    """
    train_df = train_fold_df.copy()
    val_df = val_fold_df.copy()
    
    # === Geolocation Aggregate Features (from train fold only) ===
    geo_aggs = compute_geo_aggregates(train_df)
    cy_aggs = compute_country_year_aggs(train_df)
    
    train_df = apply_geo_aggregates(train_df, geo_aggs, cy_aggs, train_df=train_fold_df)
    val_df = apply_geo_aggregates(val_df, geo_aggs, cy_aggs, train_df=train_fold_df)
    
    # === Target Encoding (from train fold only) ===
    target_encode_cols = ['geolocation_name']
    te_model = TargetEncoder(smoothing=10, min_samples=3)
    te_model.fit(train_df[target_encode_cols], train_df[TARGET_COL], cols=target_encode_cols)
    
    train_df[target_encode_cols] = te_model.transform(train_df[target_encode_cols])
    val_df[target_encode_cols] = te_model.transform(val_df[target_encode_cols])
    
    # === Label Encode Remaining Categoricals ===
    if fit_label_encoders:
        label_encoders = {}
        for col in cat_cols_to_encode:
            if col in train_df.columns:
                le = LabelEncoder()
                # Fit on combined unique values to avoid unseen category issues
                all_vals = pd.concat([train_df[col], val_df[col]]).astype(str).unique()
                le.fit(all_vals)
                train_df[col] = le.transform(train_df[col].astype(str))
                val_df[col] = le.transform(val_df[col].astype(str))
                label_encoders[col] = le
    else:
        for col in cat_cols_to_encode:
            if col in label_encoders and col in train_df.columns:
                le = label_encoders[col]
                known = set(le.classes_)
                default = le.classes_[0]
                train_df[col] = le.transform(train_df[col].astype(str).apply(lambda x: x if x in known else default))
                val_df[col] = le.transform(val_df[col].astype(str).apply(lambda x: x if x in known else default))
    
    # === Spatial KNN Features ===
    knn_feats_train = pd.DataFrame()
    knn_feats_val = pd.DataFrame()
    
    spatial_knn = None
    if 'lat' in train_df.columns and 'lon' in train_df.columns:
        spatial_knn = SpatialKNN(n_neighbors=[5, 10, 20])
        # Fit on train fold
        spatial_knn.fit(train_df[['lat', 'lon']], train_df[TARGET_COL])
        
        # Transform train (is_train=True -> LOO)
        knn_feats_train = spatial_knn.transform(train_df[['lat', 'lon']], is_train=True)
        
        # Transform val (is_train=False)
        knn_feats_val = spatial_knn.transform(val_df[['lat', 'lon']], is_train=False)
        
        # Add to DFs
        train_df = pd.concat([train_df, knn_feats_train], axis=1)
        val_df = pd.concat([val_df, knn_feats_val], axis=1)
    
    return train_df, val_df, te_model, geo_aggs, cy_aggs, label_encoders, spatial_knn


def get_feature_cols(df):
    """Get feature columns, excluding metadata and target."""
    drop_cols = [
        'data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name',
        'quarter_label', TARGET_COL, 'geolocation_name', 'country'
    ]
    return [c for c in df.columns if c not in drop_cols and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]


def get_categorical_cols(df):
    """Identify categorical columns that still need encoding."""
    drop_cols = [
        'data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name',
        'quarter_label', TARGET_COL, 'geolocation_name'
    ]
    return [c for c in df.columns
            if (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == 'object')
            and c not in drop_cols]


# =================================================================================================
# METRIC
# =================================================================================================
def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error."""
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# =================================================================================================
# MODEL DEFINITIONS
# =================================================================================================
def get_optimized_params(country, model_type, seed=42):
    """Return country-specific optimized hyperparameters."""
    # Default / Japan params (Base)
    params = {}
    
    if model_type == 'xgb':
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 5000,
            'learning_rate': 0.01,
            'max_depth': 7,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.1,
            'random_state': seed,
            'n_jobs': -1,
            'early_stopping_rounds': 100,
        }
    elif model_type == 'lgb':
        params = {
            'objective': 'regression',
            'n_estimators': 5000,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': -1,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': seed,
            'n_jobs': -1,
            'verbose': -1,
        }
    elif model_type == 'catboost':
        params = {
            'iterations': 3000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_strength': 1,
            'border_count': 254,
            'random_seed': seed,
            'verbose': 200,
            'early_stopping_rounds': 100,
        }
        
    # Philippines Specific Tuning (High Volatility, Fewer Samples)
    if country == 'Philippines':
        if model_type == 'xgb':
            # Lower capacity, higher regularization, lower LR
            params.update({
                'learning_rate': 0.005,
                'max_depth': 5,         # Shallower trees
                'min_child_weight': 10, # More conservative leaf nodes
                'reg_alpha': 0.5,       # Relaxed from 1.0
                'reg_lambda': 2.0,      # Relaxed from 5.0
                'subsample': 0.6,
                'colsample_bytree': 0.5,
            })
        elif model_type == 'lgb':
            params.update({
                'learning_rate': 0.005,
                'num_leaves': 15,       # Smaller trees
                'min_child_samples': 20, # More samples per leaf
                'reg_alpha': 1.0,
                'reg_lambda': 2.0,      # Relaxed from 5.0
                'feature_fraction': 0.6,
            })
        elif model_type == 'catboost':
            params.update({
                'learning_rate': 0.01,
                'depth': 5,
                'l2_leaf_reg': 10,      # High regularization
                'random_strength': 5,   # More randomness to avoid overfitting
            })
            
    return params


def create_xgb_model(params):
    """Create tuned XGBoost model with params."""
    # Separate kwargs for constructor vs fit/others if needed, but XGBRegressor handles most via **kwargs or set_params
    # However, early_stopping_rounds is often passed to fit(), but in sklearn API it can be in constructor for recent versions.
    # To be safe with older versions, we might need to handle it. 
    # But for simplicity, we assume standard sklearn API compatible.
    # Note: 'early_stopping_rounds' in constructor is supported in recent xgboost.
    return xgb.XGBRegressor(**params)


def create_lgb_model(params):
    """Create tuned LightGBM model with params."""
    return lgb.LGBMRegressor(**params)


def create_catboost_model(params):
    """Create tuned CatBoost model with params."""
    return cb.CatBoostRegressor(**params)


# =================================================================================================
# COUNTRY-STRATIFIED ENSEMBLE TRAINING
# =================================================================================================
def train_country_stratified_ensemble(df, base_feature_cols_hint, n_folds=5, seed=42,
                                       model_types=('xgb', 'lgb', 'catboost')):
    """
    Train separate model ensembles per country with in-fold encoding.
    
    For each country:
      For each fold:
        1. Compute geo aggregates from train fold
        2. Fit target encoder on train fold
        3. Train XGBoost, LightGBM, CatBoost on train fold
        4. Predict on val fold
    
    Returns:
        all_models: dict of {country_code: {model_type: [fold_models]}}
        oof_preds: dict of {model_type: np.array of OOF predictions}
        fold_artifacts: saved encoders/aggregates for final inference
    """
    # Store original country before any encoding
    original_country = df['country'].values.copy()
    unique_countries = np.unique(original_country)
    
    print(f"\nCountry-stratified ensemble training")
    print(f"Countries: {unique_countries}")
    print(f"Model types: {model_types}")
    print(f"Folds: {n_folds}")
    
    # Identify categorical columns to encode
    cat_cols = get_categorical_cols(df)
    # Remove 'country' — we use it for stratification, not as feature in stratified models
    cat_cols = [c for c in cat_cols if c != 'country']
    
    oof_preds = {mt: np.zeros(len(df)) for mt in model_types}
    all_models = {}
    all_fold_artifacts = {}
    all_scores = {mt: [] for mt in model_types}
    
    for country in unique_countries:
        country_mask = original_country == country
        country_df = df[country_mask].copy().reset_index(drop=True)
        country_indices = np.where(country_mask)[0]
        
        print(f"\n{'='*70}")
        print(f"Country: {country} ({country_mask.sum()} samples)")
        print(f"{'='*70}")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        country_models = {mt: [] for mt in model_types}
        country_artifacts = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(country_df)):
            print(f"\n  --- Fold {fold+1}/{n_folds} ---")
            
            train_fold = country_df.iloc[train_idx].copy()
            val_fold = country_df.iloc[val_idx].copy()
            
            # In-fold preprocessing (geo aggregates + target encoding)
            train_processed, val_processed, te_model, geo_aggs, cy_aggs, le, spatial_knn = fold_preprocess(train_fold, val_fold, cat_cols)
            
            # Get feature columns (after all preprocessing)
            feature_cols = get_feature_cols(train_processed)
            
            X_train = train_processed[feature_cols]
            y_train_log = np.log1p(train_processed[TARGET_COL])
            X_val = val_processed[feature_cols]
            y_val_orig = val_processed[TARGET_COL].values
            y_val_log = np.log1p(val_processed[TARGET_COL])
            
            # Save fold artifacts for inference
            country_artifacts.append({
                'te_model': te_model,
                'geo_aggs': geo_aggs,
                'cy_aggs': cy_aggs,
                'label_encoders': le,
                'spatial_knn': spatial_knn,
                'feature_cols': feature_cols,
            })
            
            # Train each model type
            for mt in model_types:
                # Get optimized params for this country and model type
                params = get_optimized_params(country, mt, seed=seed + fold)
                
                if mt == 'xgb':
                    model = create_xgb_model(params)
                    model.fit(
                        X_train, y_train_log,
                        eval_set=[(X_val, y_val_log)],
                        verbose=200
                    )
                    best_iter = model.best_iteration
                    
                elif mt == 'lgb':
                    model = create_lgb_model(params)
                    
                    # LGBM sklearn API uses callbacks for early stopping in recent versions
                    # We need to ensure params doesn't contain early_stopping_rounds if we pass it to fit
                    # But we passed it in create_lgb_model -> **params. 
                    # Warning: LGBMRegressor might complain if early_stopping_rounds is in **params AND callbacks.
                    # Let's clean params for LGBM if needed, but usually it handles it.
                    # Actually, we didn't put early_stopping_rounds in LGB params dict above, we used callbacks.
                    
                    model.fit(
                        X_train, y_train_log,
                        eval_set=[(X_val, y_val_log)],
                        eval_metric='rmse',
                        callbacks=[
                            lgb.early_stopping(100, verbose=False),
                            lgb.log_evaluation(200)
                        ]
                    )
                    best_iter = model.best_iteration_
                    
                elif mt == 'catboost':
                    model = create_catboost_model(params)
                    model.fit(
                        X_train, y_train_log,
                        eval_set=(X_val, y_val_log),
                        verbose=200,
                        early_stopping_rounds=200
                    )
                    best_iter = model.best_iteration_
                
                # Predict
                val_preds = np.expm1(model.predict(X_val))
                val_preds = np.maximum(val_preds, 0)
                
                # Map back to original indices
                orig_indices = country_indices[val_idx]
                oof_preds[mt][orig_indices] = val_preds
                
                score = rmsle(y_val_orig, val_preds)
                all_scores[mt].append(score)
                
                print(f"    {mt:>8s} Fold {fold+1} RMSLE: {score:.4f} (best_iter={best_iter})")
                country_models[mt].append(model)
        
        all_models[country] = country_models
        all_fold_artifacts[country] = country_artifacts
        
        # Per-country summary
        for mt in model_types:
            country_scores = all_scores[mt][-n_folds:]
            print(f"\n  {mt} Country avg RMSLE: {np.mean(country_scores):.4f} "
                  f"+/- {np.std(country_scores):.4f}")
    
    # === Overall OOF Scores ===
    print(f"\n{'='*70}")
    print("OVERALL OOF RESULTS")
    print(f"{'='*70}")
    
    y_true = df[TARGET_COL].values
    
    for mt in model_types:
        oof_score = rmsle(y_true, oof_preds[mt])
        print(f"  {mt:>8s} OOF RMSLE: {oof_score:.4f} "
              f"(mean fold: {np.mean(all_scores[mt]):.4f} +/- {np.std(all_scores[mt]):.4f})")
    
    # === Ensemble OOF ===
    # Simple average ensemble
    ensemble_preds = np.mean([oof_preds[mt] for mt in model_types], axis=0)
    ensemble_score = rmsle(y_true, ensemble_preds)
    print(f"\n  {'ENSEMBLE':>8s} OOF RMSLE: {ensemble_score:.4f}")
    
    # Optimize ensemble weights on OOF
    best_weights = optimize_ensemble_weights(
        {mt: oof_preds[mt] for mt in model_types}, y_true
    )
    weighted_preds = np.zeros(len(df))
    for mt, w in best_weights.items():
        weighted_preds += w * oof_preds[mt]
    weighted_score = rmsle(y_true, weighted_preds)
    print(f"  {'WEIGHTED':>8s} OOF RMSLE: {weighted_score:.4f}")
    print(f"  Optimal weights: {best_weights}")
    
    # === Feature Importance Plot ===
    if 'xgb' in model_types:
        last_xgb = list(all_models.values())[-1]['xgb'][-1]
        last_features = all_fold_artifacts[list(all_fold_artifacts.keys())[-1]][-1]['feature_cols']
        plt.figure(figsize=(10, 14))
        importance = last_xgb.feature_importances_
        indices = np.argsort(importance)[-35:]
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [last_features[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 35 Features (XGBoost, Last Fold)')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=100)
        print("\nSaved feature_importance.png")
    
    return all_models, oof_preds, all_fold_artifacts, best_weights


def optimize_ensemble_weights(oof_dict, y_true, n_steps=100):
    """Find optimal ensemble weights by grid search on OOF RMSLE."""
    model_types = list(oof_dict.keys())
    n_models = len(model_types)
    
    if n_models == 1:
        return {model_types[0]: 1.0}
    
    best_score = float('inf')
    best_weights = {mt: 1.0 / n_models for mt in model_types}
    
    if n_models == 2:
        for w1 in np.linspace(0, 1, n_steps + 1):
            w2 = 1.0 - w1
            preds = w1 * oof_dict[model_types[0]] + w2 * oof_dict[model_types[1]]
            score = rmsle(y_true, preds)
            if score < best_score:
                best_score = score
                best_weights = {model_types[0]: w1, model_types[1]: w2}
    elif n_models == 3:
        step = 1.0 / n_steps
        for w1 in np.arange(0, 1.0 + step / 2, step):
            for w2 in np.arange(0, 1.0 - w1 + step / 2, step):
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w3 = max(w3, 0)
                preds = (w1 * oof_dict[model_types[0]] +
                         w2 * oof_dict[model_types[1]] +
                         w3 * oof_dict[model_types[2]])
                score = rmsle(y_true, preds)
                if score < best_score:
                    best_score = score
                    best_weights = {model_types[0]: round(w1, 3),
                                    model_types[1]: round(w2, 3),
                                    model_types[2]: round(w3, 3)}
    
    return best_weights


# =================================================================================================
# MAIN
# =================================================================================================
def main(args):
    # Load
    merged_df, emb_cols = load_data(args.data_root)
    
    # Base preprocessing (PCA, time features, interactions — no leakage)
    processed_df, pca_model = base_preprocess(
        merged_df, emb_cols, pca_components=args.pca_components, seed=args.seed
    )
    
    model_types = tuple(args.models.split(','))
    print(f"\nModel types: {model_types}")
    print(f"Total samples: {len(processed_df)}")
    
    # Train ensemble
    all_models, oof_preds, fold_artifacts, best_weights = train_country_stratified_ensemble(
        processed_df, emb_cols, n_folds=args.n_folds, seed=args.seed,
        model_types=model_types
    )
    
    # === Save all artifacts ===
    joblib.dump(pca_model, 'pca_model.pkl')
    joblib.dump(all_models, 'ensemble_models.pkl')
    joblib.dump(fold_artifacts, 'fold_artifacts.pkl')
    joblib.dump({
        'model_types': model_types,
        'best_weights': best_weights,
        'n_folds': args.n_folds,
        'pca_components': args.pca_components,
    }, 'training_meta.pkl')
    
    # Also save OOF predictions for analysis
    oof_df = pd.DataFrame({'data_id': processed_df['data_id']})
    for mt in model_types:
        oof_df[f'oof_{mt}'] = oof_preds[mt]
    oof_df['oof_ensemble'] = np.zeros(len(processed_df))
    for mt, w in best_weights.items():
        oof_df['oof_ensemble'] += w * oof_preds[mt]
    oof_df.to_csv('oof_predictions.csv', index=False)
    
    print(f"\nSaved: pca_model, ensemble_models, fold_artifacts, training_meta, oof_predictions")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CCP Ensemble Model')
    parser.add_argument('--data_root', type=str, 
                        default='/content/drive/MyDrive/Competitions/Solafune/CCP/data',
                        help='Data root directory')
    parser.add_argument('--pca_components', type=int, default=64, help='Number of PCA components')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--models', type=str, default='xgb,lgb,catboost',
                        help='Comma-separated model types: xgb,lgb,catboost')
    args = parser.parse_args()
    main(args)
