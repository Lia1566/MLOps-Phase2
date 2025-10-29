#!/usr/bin/env python3
"""
Data Preparation Script

This script prepares raw data for machine learning with full preprocessing:
1. Loads raw data from data/raw/
2. Cleans column names
3. Removes duplicates
4. Creates binary target variable
5. Encodes ordinal features (Class_X_Percentage, Class_XII_Percentage, time)
6. One-hot encodes nominal features (Gender, Caste, coaching, etc.)
7. Splits into train/test sets
8. Saves fully preprocessed data to data/processed/

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config config/config.yaml
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.data.loader import load_raw_data, save_dataframe
from src.data.preprocessing import (
    clean_column_names,
    remove_duplicates,
    create_binary_target,
    encode_ordinal_features,
    encode_nominal_features,
    split_train_test
)
from src.features.engineering import get_ordinal_mappings, get_nominal_columns

# DVC Integration
try:
    from src.utils.dvc_manager import DVCManager, create_dvcignore
    DVC_AVAILABLE = True
except ImportError:
    print('Warning: DVC manager not available. Install it to enable DVC features.')
    DVC_AVAILABLE = False

def track_with_dvc(config, train_file, test_file, features_names_file, use_dvc=True):
    """
    Track processed data files with DVC for versioning. 
    
    Args:
        config: Configuration object
        train_file: Path to training data file
        test_file: Path to test data file
        features_names_file: Path to feature names file
        use_dvc: Whether to use DVC tracking
    """
    if not use_dvc or not DVC_AVAILABLE:
        return
    
    print("\n" + "="*80)
    print('DVC TRACKING')
    print("="*80)
    
    try:
        # Initialize DVC manager
        dvc = DVCManager(project_root=config.project_root)
        
        # Initialize DVC if not already done
        if not dvc.is_initialized():
            dvc.initialize()
            create_dvcignore(config.project_root)
            print("DVC initialized")
            
        print("\nTracking processed data files with DVC...")
        
        files_to_track = [train_file, test_file, features_names_file]
        
        for file_path in files_to_track:
            if file_path.exists():
                print(f" Tracking: {file_path.name}")
                success = dvc.track_file(file_path, commit=True)
                if success:
                    print(f"  {file_path.name} tracked with DVC")
                else:
                    print(f"  Warning: {file_path.name} could not be tracked with DVC")
        print(f"\nTracked {len(files_to_track)} file(s)")
        
        # Update params.yaml with preprocessing parameters
        print("\nUpdating parameters in params.yaml...")
        preprocessing_params = {
            'preprocessing': {
                'test_size': config.get('data.test_size', 0.2), 
                'random_state': config.get('data.random_state', 42), 
                'scaler': 'standard'
            }
        }
        dvc.update_params(preprocessing_params)
        print("Parameters updated")
        
        # Check remote and offer to push
        remotes = dvc.list_remotes()
        if remotes:
            print(f"\nDVC remote configured: {remotes[0]}")
            print("You can push data with: dvc push")
        else:
            print("\nNo DVC remote configured")
            print("Configure one with: dvc remote add -d storage <url>")
        
        # Show status
        print("\nDVC Status:")
        dvc.status()
        
        print("\n" + "="*80)
        print("DVC TRACKING COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\nDVC tracking failed: {e}")
        print("Continuing without DVC tracking...")

                    
def prepare_data(config, args):
    """Prepare raw data for training."""
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    # Load raw data
    print("\nLoading raw data...")
    raw_filename = 'student_entry_performance_original.csv'  
    
    try:
        df = load_raw_data(config.data_raw, raw_filename)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Looking for: {config.data_raw / raw_filename}")
        print("\nPlease make sure your data file is in data/raw/")
        sys.exit(1)
        
    # Clean column names
    print("\n" + "-"*80)
    print("CLEANING DATA")
    print("-"*80)
    df = clean_column_names(df)
    print(f"Column names cleaned")
    
    # Remove duplciates
    original_len = len(df)
    df = remove_duplicates(df)
    print(f" Duplicated removed: {original_len - len(df)} rows")
    
    # Create binary target
    print("\nCREATING BINARY TARGET")
    target_col = 'Performance'
    new_target_col = config.get('data.target_column', 'Performance_Binary')
    high_classes = ['Excellent', 'Vg']  # High performance classes
    
    df = create_binary_target(
        df,
        target_col=target_col,
        new_col=new_target_col,
        high_classes=high_classes
    )
    print(f"Binary target created: {new_target_col}")
    print(f"Class distribution:")
    print(f"- Class 0 (Lower): Class 0 (Lower): {(df[new_target_col] == 0).sum()} ({(df[new_target_col] == 0).sum()/len(df)*100:.1f}%)")
    print(f"- Class 1 (High): {(df[new_target_col] == 1).sum()} ({(df[new_target_col] == 1).sum()/len(df)*100:.1f}%)")
    
    # Encode ordinal features
    print("\n" + "-"*80)
    print("ENCODING ORDINAL FEATURES")
    print("-"*80)
    print("Ordinal features have inherent order (e.g., Good < Vg < Excellent)")
    
    ordinal_mappings = get_ordinal_mappings()
    print(f"\nOrdinal mappings:")
    for feature, mapping in ordinal_mappings.items():
        if feature in df.columns:
            print(f"{feature}:{mapping}")
    
    df, applied_mappings = encode_ordinal_features(df, ordinal_mappings)
    print(f"\nEncoded {len(applied_mappings)} ordinal features")
    
    # Drop original ordinal columns (keep only encoded versions)
    for col in ordinal_mappings.keys():
        if col in df.columns:
            df = df.drop(col, axis=1)
            print(f"Dropped original column: {col}")
        
        # Check and handle NaN values from encoding
        print(f"\n   Checking for NaN values after encoding...")
        for col in applied_mappings.keys():
            nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"   ⚠️  {col}: {nan_count} NaN values found")
            # Fill with median
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"      Filled with median value: {median_val}")
    print(f"NaN handling complete")
    
            
    # ONEHOT encode nominal feature
    print("\n" + "-"*80)
    print("ONE-HOT ENCODING NOMINAL FEATURES")
    print("-"*80)
    print("Nominal features have no inherent order (e.g., Gender, Caste)")
    
    nominal_cols = get_nominal_columns()
    # Only encode columns that exist in dataframe
    nominal_cols = [col for col in nominal_cols if col in df.columns]
    
    print(f"\nNominal features to encode:")
    for col in nominal_cols:
        unique_vals = df[col].unique()
        print(f" {col}: {len(unique_vals)} categories - {list(unique_vals)[:5]}")
        
    # Store original column count
    original_cols = len(df.columns)
    
    # One hot encode
    df = encode_nominal_features(df, nominal_cols, drop_first=True)
    
    new_cols = len(df.columns) - original_cols + len(nominal_cols) 
    print(f"\nOne Hot encoding complete")
    print(f" {len(nominal_cols)} categorical columns → {new_cols} binary features")
    print(f"Total features row: {len(df.columns)}")
    
    # Prepare features and target
    print("\n" + "-"*80)
    print("PREPARING FEATURES AND TARGET")
    print("-"*80)  
    
    # Drop original Performance column, keep binary target
    columns_to_drop = [target_col, new_target_col]
    X = df.drop(columns_to_drop, axis=1, errors='ignore')
    y = df[new_target_col]
    
    print(f"Features prepared")
    print(f"Number of features: {X.shape[1]}")
    print(f"All features are now numeric: {X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]}")
    
    # Check for any non numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric: 
        print(f"\nWarning: Found non-numeric columns: {non_numeric}")
        print("These will need to be handled before training")
    else:
        print(f"All features are numeric - ready for ML algorithms")
        
    # Display features names
    print(f"\nFeature names (first 20):")
    for i, col in enumerate(X.columns[:20], 1):
        print(f"      {i}. {col}")
    if len(X.columns) > 20:
        print(f"      ... and {len(X.columns) - 20} more")

    # Split train/test
    print("\n" + "-"*80)
    print("SPLITTING TRAIN/TEST")
    print("-"*80)
    test_size = config.get('data.test_size', 0.2)
    random_state = config.get('data.random_state', 42)
    stratify = config.get('data.stratify', True)
    
    X_train, X_test, y_train, y_test = split_train_test(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify)
    
    print(f"Data split complete")
    print(f"Train: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"Test: {len(X_test)} samples ({test_size*100:.0f}%)")
    print(f"Features: {X_train.shape[1]}")
    
    # Save processed data
    train_df = X_train.copy()
    train_df[new_target_col] = y_train.values
    
    test_df = X_test.copy()
    test_df[new_target_col] = y_test.values
    
    # Save train data
    train_filename = config.get('data.train_file', 'student_performance_train.csv')
    save_dataframe(train_df, config.data_processed, train_filename)
    print(f"Train data saved: {config.data_processed / train_filename}")
    
    # Save test data
    test_filename = config.get('data.test_file', 'student_performance_test.csv')
    save_dataframe(test_df, config.data_processed, test_filename)
    print(f"Test data saved: {config.data_processed / test_filename}")
    
    # Save feature names for reference
    feature_names_path = config.data_processed / 'feature_names.txt'
    with open(feature_names_path, 'w') as f:
        for feature in X.columns:
            f.write(f"{feature}\n")
    print(f"Feature names saved: {feature_names_path}")
    
    # Save preprocessing info
    preprocessing_info_path = config.data_processed / 'preprocessing_info.txt'
    with open(preprocessing_info_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PREPROCESSING INFORMATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("Ordinal Encoding Applied:\n")
        f.write("-"*80 + "\n")
        for feature, mapping in ordinal_mappings.items():
            if feature in ordinal_mappings:
                f.write(f"{feature}:\n")
                for value, code in mapping.items():
                    f.write(f"  {value} → {code}\n")
                f.write("\n")
        
        f.write("One-Hot Encoding Applied:\n")
        f.write("-"*80 + "\n")
        f.write(f"Original categorical features: {nominal_cols}\n")
        f.write(f"Resulting binary features: {new_cols}\n\n")
        
        f.write("Final Feature Set:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total features: {len(X.columns)}\n")
        f.write(f"All numeric: Yes\n\n")
        
        f.write("Features:\n")
        for i, col in enumerate(X.columns, 1):
            f.write(f"{i}. {col}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Preprocessing info saved: {preprocessing_info_path}")
    
    train_path = config.data_processed / train_filename
    test_path = config.data_processed / test_filename
    
    # DVC Tracking (if enabled)
    if args.track_with_dvc:
        track_with_dvc(
            config, 
            train_path,
            test_path, 
            feature_names_path, 
            use_dvc=True
        )
    
    # Verify files were created
    print("\n" + "-"*80)
    print("VERIFICATION")
    print("-"*80)
    
    if train_path.exists() and test_path.exists():
        # Verify data is numeric
        verify_train = pd.read_csv(train_path)
        verify_test = pd.read_csv(test_path)
        
        train_numeric = verify_train.drop(new_target_col, axis=1).select_dtypes(include=[np.number]).shape[1]
        train_total = verify_train.drop(new_target_col, axis=1).shape[1]
        
        print(f"All files created successfully!")
        print(f" {train_path} ({train_df.shape})")
        print(f" {test_path} ({test_df.shape})")
        print(f"All features numeric: {train_numeric}/{train_total}")
        
        if train_numeric == train_total:
            print(f"\nDATA IS READY FOR PIPELINE TRAINING")
        else:
            print(f"\nWarning: Some features are still non-numeric")
    else:
        print("Error: Files were not created properly")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\nSummary:")
    print(f"Original data: {original_len} rows, {len(df.columns)} columns")
    print(f"After preprocessing:")
    print(f" - Training: {len(train_df)} samples ({(len(train_df)/(len(train_df)+len(test_df))*100):.1f}%)")
    print(f" - Test: {len(test_df)} samples ({(len(test_df)/(len(train_df)+len(test_df))*100):.1f}%)")
    print(f" - Features: {len(X.columns)} (all numeric)")
    print(f" - Target: {new_target_col}")
    
    print(f"\nOutput files:")
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")
    print(f"Feature names: {feature_names_path}")
    print(f"Preprocessing info: {preprocessing_info_path}")
    
    if args.track_with_dvc and DVC_AVAILABLE:
        print(f"\nData files tracked with DVC")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main data preparation script."""
    parser = argparse.ArgumentParser(
        description='Comprehensive data preparation for machine learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data with default config
  python scripts/prepare_data.py
  
  # Use custom config
  python scripts/prepare_data.py --config my_config.yaml

This script performs full preprocessing:
  1. Load raw data
  2. Clean and deduplicate
  3. Create binary target
  4. Encode ordinal features (Class_X_Percentage, Class_XII_Percentage, time)
  5. One-hot encode nominal features (Gender, Caste, coaching, etc.)
  6. Split into 80% train / 20% test
  7. Save fully preprocessed numeric data
  
The output data is ready for ML pipelines and models!
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--track-with-dvc',
        action='store_true',
        help='Enable DVC tracking for data files'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(Path(args.config))
        print("\nConfiguration loaded successfully")
        print(f"Config file: {args.config}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please create a configuration file or specify --config")
        sys.exit(1)
    
    # Create necessary directories
    config.create_directories()
    print("Project directories created/verified")
    
    # Run data preparation
    prepare_data(config, args)


if __name__ == '__main__':
    main()
    
    


















