"""
AutoGluon Solution for Solafune Construction Cost Prediction

INSTRUCTIONS FOR COLAB:
1. Upload your 'data' folder to the Colab environment (e.g., /content/data).
   The structure should be:
   /content/data/train_dataset/train_tabular.csv
   /content/data/evaluation_dataset/evaluation_tabular_no_target.csv
   /content/data/sample_submission.csv

2. Install dependencies by running this in a cell:
   !pip install "autogluon[tabarena]==1.5.0"

3. Run this script.
"""

import pandas as pd
import numpy as np
import os
import argparse

# Check if AutoGluon is installed
try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except ImportError:
    print("Error: AutoGluon not installed. Run: pip install \"autogluon[tabarena]==1.5.0\"")
    raise

# --- CONFIGURATION ---
DATA_DIR = "./data" # Change this if your data is elsewhere (e.g., "/content/data")
TRAIN_PATH = os.path.join(DATA_DIR, 'train_dataset/train_tabular.csv')
TEST_PATH = os.path.join(DATA_DIR, 'evaluation_dataset/evaluation_tabular_no_target.csv')
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')
OUTPUT_FILE = 'submission_autogluon_extreme.csv'
TIME_LIMIT = 3600 # 1 hour by default, adjustable
RANDOM_STATE = 42
# ---------------------

def main():
    print("--- Starting AutoGluon Solution ---")
    
    # Load Data
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found at {TRAIN_PATH}. Please check DATA_DIR.")
        
    print(f"Loading data from {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    if os.path.exists(SAMPLE_SUB_PATH):
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    else:
        print("Warning: Sample submission not found. Creating dummy based on test IDs.")
        sample_sub = pd.DataFrame({'data_id': test_df['data_id'], 'construction_cost_per_m2_usd': 0})

    # Separate Target
    target_col = 'construction_cost_per_m2_usd'
    if target_col not in train_df.columns:
        raise ValueError(f"Target column {target_col} not found in training data.")
    
    # Log transform target for RMSLE optimization
    # AutoGluon optimizes for RMSE by default (which works for log-target to approximate RMSLE)
    # Note: AutoGluon has an 'eval_metric', we can set it to 'root_mean_squared_error'.
    print("Log-transforming target for RMSLE optimization...")
    train_df[target_col] = np.log1p(train_df[target_col])
    
    # Identify ID columns to drop from feature set
    # AutoGluon automatically handles ID columns if declared, or we can drop them.
    # explicit dropping is safer.
    id_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    
    # We leave 'geolocation_name' as is; AutoGluon handles text/categoricals well.
    
    train_data = TabularDataset(train_df)
    test_data = TabularDataset(test_df) # AutoGluon format

    # Initialize Predictor
    # path: where to save models
    save_path = 'ag_models_pricing_prediction_extreme'
    
    predictor = TabularPredictor(
        label=target_col,
        path=save_path,
        eval_metric='root_mean_squared_error', # Optimized for RMSE on log-transformed target
        problem_type='regression'
    ).fit(
        train_data,
        time_limit=TIME_LIMIT,
        presets='extreme', # High quality: stacking, bagging
        excluded_model_types=None,
        num_bag_folds=5, # 5-fold bagging
        num_bag_sets=1, # 1 set of 5-fold bagging is enough typically
        num_stack_levels=1, # 1 level of stacking
        keep_only_best=True, # Save disk space
        save_space=True,
        verbosity=2
    )

    # Summary
    print("\nTraining complete. Leaderboard:")
    print(predictor.leaderboard(silent=True))

    # Prediction on Test Data
    print("\nPredicting on test data...")
    # Make sure to drop ID columns if they exist in test data so model doesn't use them (though fit usually handles it)
    # The predictor expects the same columns as train.
    # train_df had ID columns removed? No, we didn't explicitly drop them from dataframe, 
    # but AutoGluon might infer them as ID if they are unique string.
    # Let's check: 'data_id' is unique.
    # Safe bet: pass test data as is, AutoGluon is robust.
    
    y_pred_log = predictor.predict(test_data)
    
    # Inverse Transform
    final_preds = np.expm1(y_pred_log)

    # Create Submission
    submission = sample_sub.copy()
    submission['construction_cost_per_m2_usd'] = final_preds
    
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
