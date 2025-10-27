# MLOps-Phase2

**Student Performance Prediction - Refactorization and Pipelines**

This project implemets a machine learning pipeline to predict student performance using various classification algorithms. The project demonstrates code refactorization, modular design, pipeline implementation, and experiment tracking with MLflow. 

## Table of Contents
- Project Overview
- Project Structure
- Features
- Installation
- Usage
- Results
- Techonology Used
- Contributions

## Project Overview

This project predicts whether students will achieve high performance (Excellent/Very Good) or lower performance based on various features including:
- Demographics (Gender, Caste)
- Academic background (Class X & XII percentages)
- Study factors (Coaching, Study time. Medium of instruction)
- Parental education and occupation

The project focuses on:
1. **Code Refactorization**: Transforming monolithic notebooks into modular, reusable code.
2. **Pipeline Implementation**: Using Scikit-learn pipelines for reproducible ML workflows.
3. **Experiment Tracking**: Comprehensive loggong with MLflow.

## Project Structure

```
Fase 2/
│
├── config/
│   └── config.yaml              # Project configuration
│
├── data/
│   ├── raw/                     # Original datasets
│   └── processed/               # Preprocessed datasets
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Data loading functions
│   │   └── preprocessing.py    # Data preprocessing functions
│   │
│   ├── features/
│   │   └── engineering.py      # Feature engineering functions
│   │
│   ├── models/
│   │   ├── train.py           # Model training functions
│   │   ├── evaluate.py        # Model evaluation functions
│   │   └── pipeline.py        # Scikit-learn pipeline implementation
│   │
│   ├── visualization/
│   │   └── plots.py           # Visualization functions
│   │
│   └── utils/
│       └── config.py          # Configuration management
│
├── scripts/
│   ├── prepare_data.py        # Data preparation pipeline
│   ├── train_pipeline.py      # Train models with pipelines
│   ├── train_model.py         # Train baseline models
│   └── evaluate_model.py      # Model evaluation script
│
├── models/                     # Saved trained models (*.pkl)
├── reports/                    # Results and documentation
│   ├── figures/               # Visualization outputs
│   └── *.csv                  # Results tables
│
├── mlruns/                     # MLflow experiment tracking
├── logs/                       # Training logs
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Features

1. **Modular Code Architecture**
   - Data Module: Data loading, preprocessing and feature engineering
   - Modles Module: Training, evaluation and pipeline implementation
   - Visualization Module: Plotting and result visualization
   - Utils Module: Configuration managemnet and helpers
  
2. Scikit-learn Pipelines
   - Automated preprocessing with StandardScalar
   - Prevents data leakage
   - Ensures reproducibility
   - Single object for deployment

3. MLflow Experiment Tracking
   - Automatic logging of parameters and metrics
   - Model versioning and storage
   - Visual comparison of experiments
   - Reproducible results
  
4. Comprehensive Data Preprocessing
   - Ordinal encoding for ordered categories
   - One-hot encoding for nominal features
   - Missing value handling
   - Train-test splitting with stratification
  
5. Multiple ML Algorithms
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors
   - Decision Tree

## Installation

### Prerequisite
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone repository
```
git clone https://github.com/Lia1566/MLOps-Phase2.git
cd Phase 2
```
2. Create vistual environment (recommended)
```
python -m venv venv
source venv/bin/activate # on windows: venv\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Create necessary directories
```
mkdir -p data/raw data/processed moodels reports/figures logs mlruns
```
5. Add data
- Place raw data file in `data/raw`
- Named: `student_entry_performance_original.csv`

## Usage

### Step 1. Prepare Data

Preprocess raw data with encoding and splitting:
```
python scripts/prepare_data.py
```
This will:
- Clean and deduplicate data
- Create binary target variable
- Encode ordinal features (grades, time)
- One-hot encode categorical features
- Split into train/test sets (80/20)
- Save processed data to `data/processed/`

### Step 2. Train Models with Pipelines
Train all baseline models with Scikit-learn pipelines:
```
python scripts/train_pipeline.py
```
This will:
- Train 6 different algorithms
- Apply StandardScaler preprocessing
- Perform 5-fold cross validation
- Log experiments to MLflow
- Save all trained pipelines to `models/`
- Generate comparison visualizations

### Step 3. View MLflow UI
Explore experiments and compare models:
```
mlflow ui
```
Then open browser to: `http://127.0.0.1:5000`

### Step 4. Evaluate Best Model
Evaluate a specific model in detal:
```
python scripts/evaluate_model.py --model models/best_pipeline_baseline.pkl
```
This will create:
- Confusion matrix
- ROC and PR curves
- Feature importance
- Classification report
- Model card

## Results

### Model Performance Comparison

| Model | Test Acurracy | Test F1 | Test Precision | Test Recall | CV Score
|-----------|-----------|-----------| -----------| -----------| -----------|
| Logistic Regression | 0.696 | 0.642 | 0.667 | 0.618 | 0.714 |
| Random Forest| 0.696 | 0.648 | 0.660 | 0.636 | 0.640 |
| SVM | 0.696 | 0.635 | 0.673 | 0.600 | 0.680 | 
| Gradient Boosting | 0.664 | 0.632 | 0.610 | 0.655 | 0.676 |
| K-Nearest Neighbors | 0.680 | 0.643 | 0.632 | 0.655 | 0.625 |
| Decision Tree | 0.624 | 0.552 | 0.580 | 0.527 | 0.602 |

**Best Model: Logistic Regression**
- Best Accuracy: 69.6%
- F1-Score: 64.2%
- Cross-Validation: 7104% (±5.4%)
- Training Time: 12.4 seconds

### Key Findings
1. Three models (Logistic Regression, Random Forest, SVM) achieved identical test accuracy (69.6%)
2. Logistic Regression won due to highest CV score and best generalization
3. Decision Tree showed signs of overfitting with lowest performance
4. All top models show balanced precision-recall trade offs

## Technologies Used
**Core Libraries**
- Python 3.12
- scikit-learn 1.3+: Machine learning algorithms and pipelines
- pandas: Data manipulation
- numpy: Numerical operations

**Experiment Tracking**
- MLflow: Experiment tracking and model versioning

**Visualization**
-matplotlib: Statistic plot
seaborn: Statistical visualization

**Configuration**
- PyYAML: Configuration file management

**Additional tool**
- joblib: Model serialization
- pathlib: Path handling

## Configuration
Project setting are managed in `config/config.yaml`

```
data:
  target_column: Performance_Binary
  test_size: 0.2
  random_state: 42

training:
  cv_folds: 5
  n_jobs: -1

mlflow:
  experiment_name: student_performance_baseline_pipeline
  tracking_uri: None # Uses local mlruns directory

```
MOdify this code to adjust project behaviour without changing code.

## Contributors

- Esteban Hidekel Solares Orozco - DevOps
- Jesús Antonio López Wayas - Software Enginner
- Natalia Nevarez Tinoco - Data Engineer
- Roberto López Baldomero - ML Engineer
- Yander Alec Ortega Rosales - Data Scientist

## Licence
This project is part of an academic assignment

## Acknowledgments

- Data source: https://archive.ics.uci.edu/dataset/582/student+performance+on+an+entrance+examination
- Course: MLOps
- Institution: Tecnologico de Monterrey






























