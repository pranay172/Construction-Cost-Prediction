# Running the competition experiments

[← README](../README.md)

Run all commands **from the repository root**, in a Python environment with the dependencies for your chosen route. The repository contains scripts and a saved feature-importance plot, but no competition data, checkpoints, fitted models, or requirements file. The commands below document the current code; they are not a guarantee of reproducing the historical scores with newer packages.

## Data layout

Download the data from the [competition page](https://community.solafune.com/competitions/1918ccd7-eb06-4cfc-822f-a9823c63b2c1?menu=about&tab=overview) and arrange it as follows. The data is provided under the competition's terms: don't redistribute it or anything derived from it (features, models, predictions), and delete it once you no longer need it for the competition.

```text
data/
├── train_dataset/
│   ├── train_tabular.csv
│   └── train_composite/                 # Sentinel-2 and VIIRS files
├── evaluation_dataset/
│   ├── evaluation_tabular_no_target.csv
│   └── evaluation_composite/            # Sentinel-2 and VIIRS files
└── sample_submission.csv
```

The filenames in the CSVs must resolve inside their split's composite directory. The AutoGluon route only needs the CSVs.

## Tabular baseline

```bash
python -m pip install "autogluon[tabarena]==1.5.0"
python solution_autogluon.py
```

The script uses `./data`, a 3,600-second training budget, the `extreme` preset, five bagging folds, and one stacking level. It has no command-line configuration; paths and the time budget are constants near the top of the file.

The `extreme` preset relies on tabular foundation models (such as TabPFN, TabDPT, TabM, and Mitra) that a plain `autogluon.tabular` install does not include, hence the `tabarena` extra. Those models are meant for a GPU and download their weights from Hugging Face on first use. Some are gated: in my run, a TabPFN-2.5 configuration failed and AutoGluon skipped it, so check the training log for skipped models.

What I actually ran: Python 3.12 and AutoGluon 1.5.0 on Google Colab with a T4 GPU, with `TIME_LIMIT = 7200` rather than the script's 3,600 seconds. Because the preset fits models with a time budget, expect different models and scores between runs and hardware.

Outputs in the working directory:

| Output | Purpose |
| --- | --- |
| `ag_models_pricing_prediction_extreme/` | AutoGluon predictor artifacts |
| `submission_autogluon_extreme.csv` | Evaluation predictions in USD/m² |

The script passes all supplied feature columns to AutoGluon. Although it declares an `id_cols` list, it does not actually drop those columns. If the sample submission is absent, it constructs an output frame from the evaluation IDs. Predictions use `expm1` without explicit nonnegative clipping, so inspect the generated CSV before submitting.

## Satellite features & ensemble

### 1. Install dependencies and supply Prithvi

The top-level scripts use:

```bash
python -m pip install numpy pandas torch rasterio tqdm scikit-learn scipy matplotlib joblib xgboost lightgbm catboost
```

All three boosting libraries are imported by the training module, even when `--models` selects a subset. Use a pandas release supporting `DataFrameGroupBy.apply(include_groups=False)`, which the training script calls.

Supply the **Prithvi EO 2.0 300M temporal-location** checkpoint and its compatible Python implementation. Downloading the [Hugging Face repository](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL) gives you both:

```text
models/
└── Prithvi-EO-2.0-300M-TL/
    ├── Prithvi_EO_V2_300M_TL.pt
    ├── prithvi_mae.py
    └── ...                             # Supporting modules required by that implementation
```

The extractor adds this directory to the Python import path and imports `PrithviMAE` from `prithvi_mae`; a weights file alone is insufficient. Install any additional dependencies required by that implementation. It must accept the configuration used in `load_prithvi_model()`.

**Check that the checkpoint loads.** If it is missing, the current script prints a warning and continues with randomly initialized weights. It also loads with `strict=False` after removing positional-embedding keys for the single-frame setup. CUDA is used when available, otherwise extraction runs on CPU.

### 2. Extract both splits

```bash
python extract_features.py --split train --data_root ./data --models_root ./models
python extract_features.py --split eval --data_root ./data --models_root ./models
```

Optional extraction flags are `--batch_size` (default `4`) and `--num_workers` (default `2`). Explicit paths override the script's original Colab defaults.

For each split, extraction writes these files to the **current working directory**, not inside `data/`:

```text
prithvi_embeddings_train.npy
prithvi_embeddings_ids_train.npy
handcrafted_features_train.csv
prithvi_embeddings_eval.npy
prithvi_embeddings_ids_eval.npy
handcrafted_features_eval.csv
```

Sentinel-2 bands B2, B3, B4, B8A, B11, and B12 are normalized and resized to 224 × 224 for a single-frame input. Patch embeddings are pooled to 1,024 dimensions. Handcrafted features summarize bands, NDVI/NDBI/NDWI, and VIIRS radiance.

Review extraction logs: sample errors can produce zero-filled image inputs instead of stopping the run. The code assumes raster coordinates are already longitude/latitude and does not reproject them.

### 3. Train the country-specific ensembles

```bash
python train_model.py --data_root ./data --pca_components 64 --n_folds 5 --seed 42 --models xgb,lgb,catboost
```

Training loads the training embeddings and ID array from the working directory, reduces embeddings with PCA, and combines them with tabular features and available handcrafted features. It fits models separately for each country, then selects model-family blend weights using OOF predictions.

| Output | Purpose |
| --- | --- |
| `pca_model.pkl` | Fitted embedding projection |
| `ensemble_models.pkl` | Country/model/fold estimators |
| `fold_artifacts.pkl` | Fold preprocessing, aggregates, encoders, and spatial neighbors |
| `training_meta.pkl` | Model names, blend weights, fold count, and PCA configuration |
| `oof_predictions.csv` | Training IDs and out-of-fold predictions |
| `feature_importance.png` | XGBoost diagnostic, when XGBoost is included |

The image output overwrites the tracked historical plot when training runs from the repository root.

### 4. Generate a submission

Keep the saved training artifacts, training inputs, and both splits' extracted features available in the same locations, then run:

```bash
python generate_submission.py --data_root ./data --output submission.csv
```

Inference reuses saved fold artifacts, averages fold predictions, and applies the learned model-family weights. Predictions are converted back with `expm1` and clipped to be nonnegative. The output columns are `data_id` and `construction_cost_per_m2_usd`.

Check the row count, IDs, finite values, and prediction distribution against the evaluation CSV and sample submission. Review warnings about missing country models: the current script can leave those rows with zero predictions.

## Interpreting and preserving a run

- **Validation scope:** five shuffled folds within each country test held-out rows, not unseen regions or future periods. Target aggregates are built from fold training rows; PCA is fitted on all training embeddings before cross-validation.
- **Blend selection:** reported ensemble OOF performance uses the same predictions that selected the weights. It is not an additional held-out estimate.
- **Historical evidence:** scores in the README come from the original project notes and linked leaderboard. Original run logs, submission CSVs, and fitted artifacts are not tracked, and the competition rules don't allow publishing them.
- **Generated files:** `data/` and `models/` are ignored by Git, but several root-level outputs above are not. Check `git status` before committing after a run so model artifacts and submissions do not slip into a documentation commit.
