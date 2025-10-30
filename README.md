# MLOps-Phase2

**Student Performance Prediction - Refactorization and Pipelines**

This project implemets a machine learning pipeline to predict student performance using various classification algorithms. The project demonstrates code refactorization, modular design, sklearn pipeline implementation, and experiment tracking with MLflow and DVC. 

## Table of Contents
- Project Overview
- Project Structure
- Features
- Team & Roles
- Installation
- Usage
- DVC Pipieline
- AWS S3 Remote Storage 
- Results
- MLflow Experiment Tracking
- Techonology Used
- Configuration
- Contributions
- Acknowledgements

## Project Overview

This project predicts whether students will achieve high performance (Excellent/Very Good) or lower performance based on various features including:
- Demographics (Gender, Caste)
- Academic background (Class X & XII percentages)
- Study factors (Coaching, Study time. Medium of instruction)
- Parental education and occupation

The project focuses on:
1. **Code Refactorization**: Transforming monolithic notebooks into modular, reusable code.
2. **Pipeline Implementation**: Using Scikit-learn pipelines for reproducible ML workflows.
3. **Experiment Tracking**: Comprehensive loggong with MLflow and DVC.
4. **Model Versioning**: Track all models with versions, parameters, and metrics.
5. **Reproducibility**: Ensure anyone can replicate experiments from scratch. 

## Project Structure

```
Fase 2/
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
├── models/                     # Saved trained models (*.pkl, DVC tracked)
├── reports/                    # Results and documentation
│   ├── figures/               # Visualization outputs
│   ├── baseline_results.csv    # Baseline model results
│   ├── pipeline_baseline_results.csv  # Pipeline model results
│   ├── tuning_results.csv      # Hyperparameter tuning results
│   └── *.txt                  # Additional documentation
│
├── mlruns/                     # MLflow experiment tracking
├── logs/                       # Training logs
│
├── dvc.yaml                       # DVC pipeline definition
├── dvc.lock                       # DVC pipeline lock file
├── params.yaml                    # DVC tracked parameters
├── .dvc/                          # DVC configuration
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
   - All preprocessing steps bundled with model
  
3. DVC Pipeline Automation
   - Complete 4-sate pipeline: `prepare_data → train_baseline → train_pipeline → train_tuning`
   - Automatic dependency tracking
   - Reproducible experiments with `dvc repro`
   - Data and model versioning
   - Pipeline visualization with `dvc dag` 

4. MLflow Experiment Tracking
   - Automatic logging of parameters and metrics
   - Model versioning and storage
   - Visual comparison of experiments
   - Reproducible results
   - Model registry for production deployment
  
5. Comprehensive Data Preprocessing
   - Ordinal encoding for ordered categories (grades, times)
   - One-hot encoding for nominal features (gender, caste, etc.)
   - Missing value handling with median imputation
   - Train-test splitting with stratification
   - Feature scaling with StandardScaler
  
6. Multiple ML Algorithms
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors
   - Decision Tree
  
7. Robust Path Management
   - Cross-platform compatibility using `pathlib.Path`
   - Centralized path configuration in `config/config.yaml`
   - Relative paths from project root (works anywhere)
   - Automatic directory creation on first run
   - No hardcoded paths in any scripts
   - Works seamlessly on Windows, Mac and Linux
   - Benefits:
        - Team members can run project without path adjustments
        - Works in Docker and CI/CD pipelines
        - Easy to move or deploy project
        - Professional software engineering practice
  
## Team & Roles

**Team 16 Members**
| Name | Role | Responsabilities | 
|-----------|-----------|-----------| 
| Estaban Hidekel Solares Orozco | DevOps Engineer | CI/CD setup, infrastructure management, DVC configuration, version control |
| Jesús Antonio López Wayas | Software Engineer | Code refactorization, modular architecture, code quality, testing | 
| Natalia Nevarez Tinoco | Data Enfineer | Data pipeline, preprocessing, feature engineering, data quality |
| Roberto López Baldomero | ML Engineer | Model training, hyperparameter tuning, pipeline implementation, MLflow setup | 
| Yander Alec Ortega Rosales | Data Scientist | EDA, model selection, evaluation metrics, results analysis | 

### Role Activities in Phase 2

**DevOps Enginer** (Esteban Hidekel)
- Set up DVC for data and model versioning
- Configured `dvc.yaml` pipeline with 4 stages
- Integrated DVC with Git workflow
- Created `dvc_manager.py` utility for automation
- Ensured reproducibility across envieronments

**Software Engineer** (Jesús Antonio)
- Refactored monolothic notebook code into modular structure
- Created reusable functions in `scr/` modules
- Implemented proper error handling and logging
- Applied PEP 8 coding standards
- Created command-line interfaces for scripts
- Maintained code documentation

**Data Engineer** (Natalia Nevarez)
- Designed and implemented data preprocessing pipeline
- Created `prepare_data.py` script with full automation
- Implemented ordinal and one-hot encoding strategies
- Handled missing values and data quality issues
- Set up data versioning with DVC
- Documented preprocessing steps
- Managed GitHub repository and version control

**ML Engineer** (Roberto López)
- Implemented Scikit-Learn pipelines in `pipeline.py`
- Set up MLflow experiment tracking
- Created training scripts (`train_model.py`, `train_pipeline.py`)
- Performed hyperparameters tuning with GridSearchCV
- Integrated models with DVC versioning
- Ensured model reproducbility

**Data Scientist** (Ynader Alec)
- Conduced exploratory data anylsis
- Selected appropriate ML algorithms
- Defined evaluation metrics and thresholds
- Analyzed model results and created visualizations
- Compared baseline vs. pipeline vs. tuned models
- Documented findings and recommendations

### Team Interactions & MLOps Workflow
```
Data Scientist → Data Engineer → ML Engineer → DevOps → Software Engineer
     ↓               ↓               ↓            ↓            ↓
   EDA & Model    Data Pipeline   Training &   DVC Setup   Code Quality
   Selection      & Features      MLflow      & Versioning  & Refactor
                                                             
                  ← Feedback Loop & Iterations →

```
### Collaborative Workflow:
1. Data Scientist analyzes data and selects models
2. Data Enfineer builds preprocessing pipeline
3. ML Engineer implements training and experiment tracking
4. Software Engineer refactors code into modular structure
5. DevOps Engineer sets up versioning and reproducibility
6. Iteration: Team reviews results and improves together


## Installation

### Prerequisite
- Python 3.8 or higher
- pip package manager
- Git
- DVC (Data Version Control)

### Setup

1. Clone repository
```
git clone https://github.com/Lia1566/MLOps-Phase2.git
cd Phase 2
```
2. Create vistual environment (recommended)
```
python -m venv venv
source venv/bin/activate # On windows: venv\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Initialize DVC (if not already initialized)
```
dvc init
```
5. Pull data and modules (if using DVC remote)
```
dvc pull
```
6. Create necessary directories
```
mkdir -p data/raw data/processed moodels reports/figures logs mlruns
```
5. Add data
- Place raw data file in `data/raw`
- File name: `student_entry_performance_original.csv`

## Usage

### Quick Start: Run Complete Pipelines
```
# Run all steps automatically
dvc repro
```
This single command will:
1. Prepare data (if needed)
2. Train baseline models (if needed)
3. Train pipeline models (if needed)
4. Perform hyperparameters tuning (if needed)

### Step-by-Step Usage

#### Step 1. Prepare Data

Preprocess raw data with encoding and splitting:
```
python scripts/prepare_data.py
```
With DVC tracking:
```
python scripts/prepare_data.py --track-with-dvc
```

Output:

- Cleans and deduplicates data
- Creates binary target variable
- Encode ordinal features (grades, study time)
- One-hot encodes categorical features
- Splits into train/test sets (80/20)
- Save processed data to `data/processed/`

#### Step 2. Train Baseline Models
Train models without pipelines:
```
python scripts/train_model.py --mode baseline
```
With DVC tracking:
```
python scripts/train_model.py --mode baseline --track-with-dvc
```
Output:
- Trains 6 different algorithms
- Performs 5 fold cross-validation
- Logs experiments to MLflow
- Saves all models to `models/`
- Generates comparison visualization

#### Step. 3 Train Models with Sklearn Pipelines
Train models with preprocessing pipelines:
```
python scripts/train_pipeline.py
```
With DVC tracking:
```
python scripts/train_pipeline.py --track-with-dvc
```
Output:
- Trains 6 different with StandardScaler preprocessing
- Each pipeline is a sing;e deployable object
- Performs 5 fold cross-validation
- Logs to MLflow
- Saves pipeline to `models/`
- Creates visualizations and documentation

#### Step 4. Hyperparameter Tuning
Tune top 3 models:
```
python scripts/train_model.py --mode tuning --top-n 3
```
With DVC tracking:
```
python scripts/train_model.py --mode tuning --top-n 3 --track-with-dvc
```
Output:
- Selects top 3 models from baseline
- Performs GridSearchCV
- Tests multiple parameter combinations
- Saves best models
- Logs all runs to MLflow

#### Step 5. View MLflow UI

Explore experiments and compare models:
```
mlflow ui
```
Then open browser to: `http://127.0.0.1:5000`

#### Step 6. Evaluate Specific Model
Detailed evaluation of a saved model:
```
python scripts/evaluate_model.py --models/best_pipeline_baseline.pkl
```
Output:
- Confusion matrix
- ROC and PR curves
- Features importance
- Classification report
- Model card with metadata

## DVC Pipeline
**Pipeline Stages**
The project uses a 4-stages DVC pipeline for complete reproducibility:
```
data/raw.dvc
     ↓
prepare_data
     ↓
     ├─→ train_baseline → train_tuning
     └─→ train_pipeline
```
### View Pipeline
```
# Show pipeline structure
dvc dag

# Check pipeline status
dvc status

# List all stages
dvc stage list
```
### Run Pipeline

```
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro prepare_data
dvc repro train_baseline
dvc repro train_pipeline
dvc repro train_tuning
```

### Pipeline Definition (`dvc.yaml`)

Stage 1: prepare_data
- Command: `python scripts/prepare_data.py --track-with-dvc`
- Dependencies: Raw data, scripts, config
- Outputs: Processed train/test CSV files
- Parameters: test_size, random_state, stratify

Stage 2: train_baseline
- Command: `python scripts/train_model.py --mode baseline --track-with-dvc`
- Dependencies: Processed data, training scripts
- Outputs: Best baseline model, results CSV
- Metrics: baseline_results.csv

Stage 3: train_pipeline
- Command: `python scripts/train_pipeline.py --track-with-dvc`
- Dependencies: Processed data, pipelines scripts
- Outputs: Best pipeline model, results CSV
- Metrics: pipeline_baseline_results.csv

Stage 4: train_tuning
- Command: `python scripts/train_model.py --mode tuning --top-n 3 --track-with-dvc`
- Dependencies: Processed data, baseline results
- Outputs: Tuned models, tuning results CSV
- Metrics: tuning_results.csv

Benefits of DVC
- Reproducibility: Anyone can run `dvc repro` to replicate experiments
- Version Control: Track data and model versions alongside code
- Efficiency: Only re-run stages when dependencies change
- Collaboration: Share data and models without Git bloat
- Automation: Define ML pipeline as code

## AWS S3 Remote Storage

This project uses AWS S3 for remote data and model storage, enabling team collaboration and version control large files. 

### Configuration
- **S3 Bucket**: `s3://itesm-mna/202502-equipo16/`
- **Region**: `us-east-2`
- **AWS Profile**: `equipo16`

### Setup for Team Members
#### 1. Install AWS CLI

**macOS:**
```
curl: "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
rm AWSCLIV2.pkg
aws --version
```
**Windows:**
Download and install from: https://awscli.amazonaws.com/AWSCLIV2.msi

#### 2. Configure AWS Credentials

Request credenctials from TA, then run:
```
aws configure --profile equipo16
```

Enter when prompted:
- **AWS Access Key ID**: [From TA]
- **AWS Secret Access Key**: [From TA]
- **Default region name**: `us-east-2`
- **Default output format**: `json`

#### 3. Verify Access
```
aws sts get-caller-identity --profile equipo16
aws s3 ls s3://itesm-mna/202502-equipo16/ --profile equipo16
```

#### 4. Install DVC with S3
```
pip install "dvc[s3"
```

### Working with S3
**Pull data and models:**
```
dvc pull
```

**Push data and models:**
```
dvc push
```

**Check sync status:**
```
dvc status
```

### Daily Workflow
When making changes:
```
# 1. Tracking new data/models:
dvc add data/new_file.csv

# 2. Push to S3
dvc push

# 3. Commit tp Git
git add data/new_file.csv.dvc .gitignore
git commit -m "Add new data"
git push
```
When pulling teammate's changes:
```
git pull
dvc pull
```

### Important Notes
- Credentials are programmatic access only - no AWS console access.

Best practices:
- Always `dvc pull` before starting work
- Always `dvc push` after tracking new files
- Never commit AWS credentials to Git

### Troubleshooting

Error: "s3 is supported, but required 'dvc-s3'"
```
pip install "dvc[s3]"
```

Error: "SignatureDoesNotMatch"
- Verify credentials with your TA
- Reconfigure: `aws configure --profile equipo16`

Error: "AccessDenied"
- Check you're using: `--equipo equipo16`
- Verify path: `s3://itesm-mna/202502-equipo16/`


## Results

### Model Performance Comparison

#### Baseline Models (No Preprocessing Pipeline)

| Model | Test Acurracy | Test F1 | Test Precision | Test Recall | CV Score (Mean ± Std)
|-----------|-----------|-----------| -----------| -----------| -----------|
| SVM | 0.736 | 0.697 | 0.704 | 0.691 | 0.688 ± 0.061 | 
| Logistic Regression | 0.712 | 0.660 | 0.686 | 0.636 | 0.712 ± 0.060 |
| Random Forest| 0.696 | 0.648 | 0.660 | 0.636 | 0.640 ± 0.040 |
| Gradient Boosting | 0.664 | 0.632 | 0.610 | 0.655 | 0.676 ± 0.032 |
| K-Nearest Neighbors | 0.640 | 0.602 | 0.586 | 0.618 | 0.559 ± 0.025 |
| Decision Tree | 0.624 | 0.552 | 0.580 | 0.527 | 0.598 ± 0.037 |

**Best Model: SVM**
- Best Accuracy: 73.6%
- F1-Score: 69.7%
- ROC-AUC: 82.1%
- Cross-Validation: 68.8% (±6.1%)

#### Pipeline Models (With StandardScaler Preprocessing)

| Model | Test Acurracy | Test F1 | Test Precision | Test Recall | CV Score (Mean ± Std)
|-----------|-----------|-----------| -----------| -----------| -----------| 
| Logistic Regression | 0.696 | 0.642 | 0.667 | 0.618 | 0.714 ± 0.054 |
| Random Forest| 0.696 | 0.648 | 0.660 | 0.636 | 0.640 ± 0.040 |
| SVM | 0.696 | 0.635 | 0.673 | 0.600 | 0.680 ± 0.067 |
| K-Nearest Neighbors | 0.680 | 0.643 | 0.632 | 0.655 | 0.625 ± 0.058 |
| Gradient Boosting | 0.664 | 0.632 | 0.610 | 0.655 | 0.676 ± 0.032 |
| Decision Tree | 0.624 | 0.552 | 0.580 | 0.527 | 0.602 ± 0.036 |

Best Pipeline Model: Logistic Regression
- Test Accuracy: 69.6%
- F1-Score: 64.2%
- Cross Validation: 71.4% (±5.4%)
- Structure: `StandardScaler → Logistic Regression`

#### Tuned Models (Hyperparameter Optimization)

| Model | Test Acurracy | Test F1 | Best Parameters | CV Score
|-----------|-----------|-----------| -----------| -----------|
| SVM | 0.720 | 0.690 | C=1, kernel=linear, gamma=scale | 0.706 | 
| Logistic Regression | 0.704 | 0.665 | C=1, penalty=l2, solver=liblinear | 0.714 | 
| Random Forest| 0.688 | 0.652 | n_estimators=50, max_depth=5 | 0.692 | 

Best Tuned Models: SVM
- Test Accuracy: 72.0%
- F1-Score: 69.0%
- Parameters: Linear kernel, C=1

### Understanding the Three Approaches

**Why do we have three different result tables?**

1. Baseline Models: Quick exploration with potential data leakage
   - Best: SVM (may be optimistic)

2. Pipeline Models: Industry best practice, no data leakage
   - Best: Logistic Regression (most reliable)
   - Recommended for deployment
  
3. Tuned Models: Hyperparameter optimization
   - Best: SVM (balanced optimization)

**Why do results vary?**
- Preprocessing strategy (leakage vs no leakage)
- Cross-Validation rigor
- Hyperparameter configuration
- Small test set (125 samples)

**Which is most important?**
For MLOps, **Pipeline Models** are most valuable because they're:
- Production-ready (single deployable object)
- Reproducible (no manual preprocessing needed)
- Best practice (prevents data leakage)

## Key Findings

1. **Baseline SVM** achieved highest test accuracy (73.6%) without pipeline preprocessing
2. Three models tied in pipeline comparison (Logistics Regression, Random Forest, SVM at 69.6%)
3. **Logistic Regression** won pipeline comparison due to highest CV score and best generalization
4. Hyperparamater tuning improved SVM to 72.0% accuracy
5. Preprocessing pipelines show trade-off between accuracy and deployment simplicity
6. All top models show balanced precision-recall trade-offs
7. Decision Tree consistently showed lowest performance, indicating need for ensemble methods

### Recommendations
- For deployment: Use Logistic Regression pipeline (simple, interpretable, reproducible)
- For highest accuracy: Use Baseline SVM (73.6% test accuracy)
- For production: Consider ensemble of top 3 models
- Next steps: Feature engineering, more data collection, deep learning exploration

## MLflow Experiment Tracking
###View Experiments
```
# Start MLflow UI
mlflow ui

# Navigate to
http://127.0.0.1:5000
```

### What's Tracked
For each experiment, MLflow logs:
- Parameters: All hyperparamters and configuration
- Metrics: Accuracy, precision, recall, F1, ROC-AUC
- Artifacts: Models, plots, confusion matrices
- Tags: Model type, experiment name, timestamp
- Environment: Python version, library versions

### Experiment Organization
- Baseline Experiment: `student_performance_baseline`
- Pipeline Experiment: `student_performance_baseline_pipeline`
- Tuning Experiment: `student_performance_tuning`

### Model Registry
All models are registered in MLflow with
- Version number
- Hyperparameters
- Training metrics
- Test metrics
- Model artifacts (.pkl) files
- Git commit hash
- Timestamp

## Technologies Used
**Core Libraries**
- Python 3.12 (Programming language)
- scikit-learn 1.3+: Machine learning algorithms and pipelines
- pandas: Data manipulation
- numpy: Numerical operations

**MLOps Tools**
- MLflow: Experiment Tracking and model versioning
- DVC: Data and pipeline versioning
- Git: Source code version control

**Visualization**
-matplotlib: Statistic plot
seaborn: Statistical visualization

**Configuration & Utilities**
- PyYAML: Configuration file management
- joblib: Model serialization
- pathlib: Path handling

**Additional Tools**
- argparse: Command-line interfaces
- logging: Application logging
- pickle: Object serialization

## Configuration
Project setting are managed in `config/config.yaml`

```
# Project metadata
project:
  name: student_performance_prediction
  version: 1.0.0

# Data configuration
data:
  target_column: Performance_Binary
  test_size: 0.2
  random_state: 42
  stratify: true

# Training configuration
training:
  cv_folds: 5
  random_state: 42
  n_jobs: -1
  baseline_models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - SVM
    - K-Nearest Neighbors
    - Decision Tree
  top_n_models: 3

# MLflow configuration
mlflow:
  baseline_experiment: student_performance_baseline
  tuning_experiment: student_performance_tuning
  tracking_uri: null  # Uses local mlruns directory
  log_models: true
  log_metrics: true
  log_params: true

# DVC configuration
dvc:
  remote:
    name: myremote
    url: null  # Set to your remote storage
  track:
    data: true
    models: true
    pipelines: true
  versioning:
    create_tags: true
    tag_prefix: v

```
MOdify this code to adjust project behaviour without changing code.

## Contributing

This project is part of an academic assignment for the MLOps course at Tecnológico de Monterrey.

### Team Contributors
All team members contributed equally to different aspects of the project following MLOps best practices and role responsabilities. 

### Git Workflow

- Main branch: `main`
- Feature branches: `features/feature-name`
- Commit message format: `type: description`
- All changes reviewd and merged via pull request

## Acknowledgments

- Data source: https://archive.ics.uci.edu/dataset/582/student+performance+on+an+entrance+examination
- Course: MLOps
- Institution: Tecnologico de Monterrey
- Course instructors and TA
- Semester: Fall 2025

### References
- Cookiecutter Data Science Template - https://cookiecutter-data-science.drivendata.org
- MLflow Documentation - https://mlflow.org/docs/latest/index.html
- DVC Documentation - https://dvc.org/doc
- Scikit-learn Pipelines - https://scikit-learn.org/stable/modules/compose.html






























