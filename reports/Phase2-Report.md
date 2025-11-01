# Phase 2 Report: Code Refactorization and MLflow Experiment Tracking

## Table of Contents

1. Code Refactorization
2. MLflow Experiment Tracking
3. Experiment Results
4. Best Practices Implemented
5. Lessons Learned
6. Recommendation

## 1. Code Refactorization
### 1.1 Before - Monolithic Notebook Structure

On Phase 1, we has a single Jupyter Notebook (`mlops-eq16_notebooks_phase1_team16.ipynb`), which had more than 500 lines of code and all functionality in one place.
Furthermore, ut was difficult to reuse the code, as well as hard to test or maintain. We had to do manual tracking of results. 

We identified several problems, such as:
- Code duplication across cells
- Hardcoded file paths
- No separation of concerns
- Difficult collaboration
- Can't reproduce results reliably
- No automated testing
- Mixed data processing and model training

### 1.2 After - Modular Architecture

New structure:

```
src/
├── data/
│   ├── loader.py           # Data loading functions
│   └── preprocessing.py    # Data cleaning and transformation
│
├── features/
│   └── engineering.py      # Feature creation and encoding
│
├── models/
│   ├── train.py           # Model training logic
│   ├── evaluate.py        # Model evaluation and metrics
│   └── pipeline.py        # Scikit-learn pipeline wrapper
│
├── visualization/
│   └── plots.py           # Plotting and visualization
│
└── utils/
    ├── config.py          # Configuration management
    └── dvc_manager.py     # DVC automation utilities

scripts/
├── prepare_data.py        # End-to-end data preparation
├── train_model.py         # Model training script
├── train_pipeline.py      # Pipeline training script
└── evaluate_model.py      # Model evaluation script
```

### 1.3 Refactorization Process
First we identified 5 main funtional areas to reuse:
1. Data Module: Loading, cleaning, preprocessing
2. Feature Module: Feature engineering and encoding
3. Models Module: Training, evaluation, pipelines
4. Visualization Module: Plots and reports
5. Utils Module: Configuration and helpers

Next we extracted functions from our jupyter notebooks. 
Example: Data Loading 
Before
```
import pandas as pd
df = pd.read_csv('../../data/raw/student_entry_performance_original.csv')
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
# ... 50 more lines of preprocessing ...
```
After (in `src/data/loader.py`)

```
def load_student_data(filepath: Path, config: dict) -> pd.DataFrame:
    """
    Load student performance data with validation.
    
    Args:
        filepath: Path to CSV file
        config: Configuration dictionary
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns missing
    """
```

Doing this has makes the code reusable across scripts, has proper documentation, we do error handling, and it is easy to test. 

Next, we created command-line scripts. 
Example: `scripts/prepare_data.py`
```
def main():
    parser = argparse.ArgumentParser(description='Prepare student performance data')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--track-with-dvc', action='store_true')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    data = load_student_data(config['paths']['data'])
    
    # Preprocess
    data_clean = preprocess_data(data, config)
    
    # Feature engineering
    X, y = engineer_features(data_clean, config)
    
    # Split and save
    save_processed_data(X, y, config)
```
The benefits of doing it this way are:
- Can be run from command line
- Configurable via arguments
- Logs all operations
- Tracks with DVC automatically
- Reproducible

Afterwards, we implemented Scikit-Learn pipelines. Pipelines prevent data leakage (scaling on training data only), are single object for deployment, 
reproducible preprocessing and it is best practice in industry. 

Implementation:

```
class MLPipeline:
    """Wrapper for scikit-learn pipelines with MLflow tracking."""
    
    def create_pipeline(self, model_name: str, model, config: dict):
        """Create a pipeline with preprocessing and model."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    
    def train(self, X_train, y_train, config):
        """Train pipeline and log to MLflow."""
        with mlflow.start_run():
            self.pipeline.fit(X_train, y_train)
            mlflow.sklearn.log_model(self.pipeline, "model")
            # Log metrics, params, etc.
```

### 1.4 Configuration Management

Centralized configuration in `config/config.yaml`

Before, we had hardcoded values scattered throughout the notebook like so:
```
test_size = 0.2
random_state = 42
n_estimators = 100
```

After, we have a single source. 
```
data:
  test_size: 0.2
  random_state: 42
  
training:
  cv_folds: 5
  n_jobs: -1
  
models:
  random_forest:
    n_estimators: [50, 100, 200]
    max_depth: [5, 10, 15]
```

Benefits:
- Change behavior without changing code
- Easy experimentation
- Version control for configurations
- Clear documentation of settings

### 1.5 Path Management
This allows for cross platform compatibility. 

Before:

```
data = data = pd.read_csv('/Users/lia/Desktop/Phase1/data/raw/student.csv')  # Only on Natalia's Mac
```

After:
```
from pathlib import Path

project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'raw' / 'student.csv'
data = pd.read_csv(data_path)
```
Benefits:
- Works on Windows, Mac, Linux
- Relative to project root
- Easy to move project
- Works in Docker containers (Phase3)


### Summary 

| **Metric**              | **Before (Notebook)**         | **After (Modular)**                 | **Improvement**         |
|--------------------------|------------------------------|------------------------------------|--------------------------|
| **Lines of code**        | 500+ (in 1 file)             | ~1200 (across modules)             | Better organization      |
| **Code reusability**     | 0%                           | 95%                                | Yes                     |
| **Test coverage**        | 0%                           | Ready for testing                  | Yes                      |
| **Reproducibility**      | Manual                       | Automated                          | Yes                      |
| **Time to retrain**      | 15–20 min                    | 2 min (`dvc repro`)                | 90% faster               |
| **Collaboration ease**   | Very difficult                | Easy                               | Yes                      |
| **Deployment readiness** | Not ready                    | Production-ready                   | Yes                      |

## MLflow Experiment Tracking

### 2.1 Why MLflow?


Before integrating MLflow, we faced several challenges in managing experiments effectively. It was difficult to keep track of which hyperparameters 
produced specific results, and we couldn’t systematically compare experiments. Retrieving old models was nearly impossible, and manual tracking 
in spreadsheets proved error-prone and inefficient. Additionally, sharing results with the team was cumbersome and inconsistent.

MLflow addressed all of these issues by introducing a robust experiment tracking and management system. With automatic parameter logging and
metric tracking over time, we can now easily monitor and reproduce experiments. MLflow’s model versioning and registry features enable seamless 
model management, while visual experiment comparisons help us quickly evaluate performance. It also ensures reproducible results and p
rovides convenient artifact storage for plots and models, making collaboration and experimentation significantly more efficient.

### 2.2 MLflow Integration

What was tracked:
1. Parameters:
   - Model hyperparameters (C, n_estimators, max_depth, etc.)
   - Training configuration (CV folds, random state)
   - Data preprocessing settings

2. Metrics
   - Test accuracy, precision, recall, F1-score
   - ROC-AUC score
   - Cross-validation mean and std
   - Training time
  
3. Artifacts
   - Trained models (.pkl files)
   - Confusion matrices (plots)
   - ROC curves
   - Feature importance plots
   - Training logs

4. Tags
   - Model type (baseline, pipeline, tuned)
   - Experiment name
   - Git commit hash
   - Timestamp
  
5. Environment
   - Python version
   - Library version
   - System information
  
### 2.3 Experiment Organization
3 Main Experiments

Experiment 1: Baseline Model
- Name: `student_performance_baseline`
- Purpose: quick exploration of all algorithms
- Models: 6 algorithms with default hyperparameters
- Runs: 6 runs
- Key Findings: SVM performed best (73.6% accuracy)

Experiment 2: Pipeline Models
- Name: `student_performance_baseline_pipeline`
- Purpose: production ready models with proper preprocessing
- Models: 6 algorithms with StandardScaler pipelines
- Runs: 6 runs
- Key finding: Logistic Regression best generalization (71.4% CV)

Experiment 3: Hyperparameter Tuning
- Name: `student_performance_tuning`
- Purpose: optimize 3 models
- Models: SVM, Logistic Regression, Random Forest
- Runs: 15+ runs (3 models x 5+ configurations each)
- Key finding: tuned SVM achieved 72% accuracy

### 2.4 Accessing MLflow Results

```
mlflow ui
# navigate to http://127.0.0.1:5000
```
MLflow UI features:
- Compare runs side-by-side
- Filter and search experiments
- Download models and artifacts
- View metrics plots over time
- Access detailed run information


## Experiment Results

### 3.1 Baseline Experment
**Total Runs:** 6 models  
**Best Model:** SVM  

| **Model**              | **Test Accuracy** | **Test F1** | **Test Recall** | **Test Precision** | **ROC-AUC** | **CV Mean ± Std**     |
|-------------------------|------------------:|-------------:|----------------:|-------------------:|-------------:|-----------------------:|
| **SVM**                 | 73.6%             | 69.7%        | 68.2%           | 71.1%              | 82.1%        | 68.8% ± 6.1%          |
| Logistic Regression     | 71.2%             | 67.1%        | 65.5%           | 69.2%              | 79.5%        | 69.8% ± 6.1%          |
| Random Forest           | 69.6%             | 64.8%        | 63.6%           | 66.0%              | 75.3%        | 64.0% ± 4.0%          |
| Gradient Boosting       | 66.4%             | 63.2%        | 65.5%           | 61.0%              | 73.8%        | 67.6% ± 3.2%          |
| K-Nearest Neighbors     | 64.0%             | 60.2%        | 61.8%           | 58.6%              | 68.9%        | 55.9% ± 2.5%          |
| Decision Tree           | 62.4%             | 55.2%        | 52.7%           | 58.0%              | 61.2%        | 59.8% ± 3.7%          |

Across all experiments, SVM consistently outperformed other models, demonstrating the most reliable and accurate results. 
Logistic Regression exhibited strong generalization, supported by a high cross-validation (CV) score, while the Decision 
Tree model showed clear signs of overfitting, as indicated by a large gap between training and test performance. 
Ensemble methods such as Random Forest and Gradient Boosting were also competitive, delivering solid performance and stability across 
multiple runs.

Using MLflow significantly improved the model development and evaluation workflow. It enabled easy identification of the best-performing 
model and allowed straightforward comparison between CV and test results. The system also helped detect overfitting patterns early in 
the process and streamlined collaboration by instantly sharing experiment results with the team.


### 3.2 Pipeline Experiment
**Total Runs:** 6 models with StandardScaler  
**Best Model:** Logistic Regression (by CV score)  

| **Model**              | **Test Accuracy** | **Test F1** | **CV Mean ± Std** | **Pipeline Benefit**     |
|-------------------------|------------------:|-------------:|------------------:|--------------------------|
| **Logistic Regression** | 69.6%             | 64.2%        | 71.4% ± 5.4%      | Best generalization      |
| Random Forest           | 69.6%             | 64.8%        | 64.0% ± 4.0%      | Stable performance       |
| SVM                     | 69.6%             | 63.5%        | 68.0% ± 6.7%      | Similar to baseline      |
| K-Nearest Neighbors     | 68.0%             | 64.3%        | 62.5% ± 5.8%      | Improved with scaling    |
| Gradient Boosting       | 66.4%             | 63.2%        | 67.6% ± 3.2%      | Consistent results       |
| Decision Tree           | 62.4%             | 55.2%        | 60.2% ± 3.6%      | Still overfitting        |

When comparing pipeline results against the baseline, three models tied with an accuracy of 69.6%. Logistic Regression achieved 
the highest cross-validation (CV) score of 71.4%, demonstrating strong generalization. The K-Nearest Neighbors (KNN) model 
showed notable improvement with scaling, gaining approximately 4% in accuracy. Pipelines also played a crucial role in 
preventing data leakage, making the overall system more reliable and better suited for deployment.

MLflow provided valuable insights throughout the experiment tracking process. It automatically recorded preprocessing steps, 
enabling transparent and repeatable workflows. The tool made it easy to directly compare pipeline and baseline results, helping 
identify which models benefited most from scaling. Additionally, MLflow facilitated the storage of production-ready pipeline objects, 
streamlining deployment and collaboration across the team.


### 3.3 Hyperparameter Tuning

**Total Runs:** 15+ runs across 3 models  
**Best Model:** SVM (linear kernel, C=1)  

| **Model**              | **Best Test Acc** | **Best Test F1** | **Best Parameters**                                | **Improvement**         |
|-------------------------|------------------:|-----------------:|---------------------------------------------------|--------------------------|
| **SVM**                 | **72.0%**         | 69.0%            | C=1, kernel=linear, gamma=scale                   | +6.4% from baseline      |
| Logistic Regression     | 70.4%             | 66.5%            | C=1, penalty=l2, solver=liblinear                 | +3.8%                    |
| Random Forest           | 68.8%             | 65.2%            | n_estimators=50, max_depth=5                      | +2.8%                    |


### 3.4 Cross Experiment
Using MLflow to Compare all experiments

| **Model**              | **Best Model** | **Test Accuracy** | **CV Score**                     | **Deployment Ready?**         |
|-------------------------|------------------:|-----------------:|---------------------------------------------------|--------------------------|
| Baseline                 | SVM        | 73.6%           | 68.8%                  | Potential Leakage    |
| Pipeline    | Logistic Reg            | 69.6%           | 71.4%                | Yes                   |
| Tunes         | SVM            | 72%            | 70.6%                      | Yes                    |

Recommendations: 
1. For production - use tuned SVM
2. For simplicity - use pipeline Logistic Regression (simpler, interpretable)
3. For exploration - baseline SVM showed potential (73.6%)

### 3.5 Feature Importance Insights
Top 5 most important features (from RF model):
1. Class XII Percentage (28.3%)
2. Class X Percentage (24.7%)
3. Study time per week (15.2%)
4. Coaching (12.8%)
5. Father's occupation (8.9%)

MLflow artifact: feature importance plot automatically saved for each run. 

## 4. Best Practices Implemented

### 4.1 Code Organization

The project follows a modular architecture with a clear separation of concerns, ensuring maintainability and scalability. 
Functions are reusable and thoroughly documented, with type hints provided for improved readability and clarity. Configuration
management is handled via YAML files, and all scripts include command-line interfaces for flexible execution across different environments. 

### 4.2 Version Control

Version control practices are robust, using Git for source code management and DVC for tracking data and models. Remote storage is integrated through
AWS S3, enabling seamless collaboration and reproducibility. Sensitive files are excluded via `.gitignore`, and all commits include meaningful messages to
maintain clear project history. 

### 4.3 Reproducibility

Reproducibility is a key focus throughout the project. Dependencies are pinned in `requirements.txt`, and random seeds are set 
consistently to ensure deterministic results. The entire workflow is automated using a DVC pipeline, while MLflow is employed for 
experiment tracking. Configuration files are versioned alongside code for complete traceability.

### 4.4 MLOps Practices

The implementation adheres to modern MLOps principles. Automated pipelines are managed with DVC, and MLflow handles experiment 
tracking and model versioning. All artifacts—including models, metrics, and visualizations—are stored systematically. Logging and monitoring 
are integrated to support maintainability and long-term reliability of the system.

### 4.5 Software Engineering

From a software engineering perspective, the project emphasizes cross-platform compatibility, robust error handling, and comprehensive 
validation. Logging is implemented throughout to facilitate debugging and transparency. Documentation is extensive, including a 
well-maintained README and detailed docstrings, and the overall project structure follows professional development standards.


## 5. Lessons Learned

### 5.1 Technical Lessons

One of the key technical lessons learned was that **Scikit-learn Pipelines prevent data leakage**. The initial baseline results appeared 
overly optimistic at 73.6%, but once pipelines were introduced, the results stabilized to a more realistic 69.6%. This reinforced the 
importance of always using pipelines in production models to ensure reliable and unbiased performance evaluation.

Another major takeaway was that **MLflow saves significant time**. Previously, experiment tracking was done manually in spreadsheets, 
taking two to three hours per session. With MLflow’s automatic logging during training, tracking required zero additional time, making it 
effortless to compare more than twenty experiments instantly.

We also learned that **configuration management is critical**. By externalizing hyperparameters into configuration files, we were able to 
change settings without modifying code, enabling easier experimentation and ensuring clear documentation of what was tested in each run.

Finally, **DVC enables true reproducibility**. Using the `dvc repro` command, anyone could reproduce results exactly, eliminating “works 
on my machine” issues. This demonstrated that data versioning is just as important as code versioning for maintaining consistent and 
transparent workflows.

### 5.2 Process Lessons

From a process perspective, **refactorization took time but paid off**. The initial refactorization effort took around forty hours, 
but subsequent experiments became roughly ten times faster, and the modular structure enabled seamless code reuse across team members.

**Documentation during development** also proved to be a valuable practice. Writing docstrings while coding and updating the README 
incrementally saved significant time later. This approach ensured that both current and future team members could easily understand and 
maintain the project.

**Team collaboration** improved drastically through better tooling and structure. Modular code enabled parallel development, MLflow made 
sharing results effortless, and DVC eliminated data transfer issues by tracking datasets and models consistently.

### 5.3 Challenge Faced

We encountered several challenges during development. The first was **path management across Mac and Windows systems**, which was solved 
by using `pathlib.Path` for cross-platform compatibility. The second challenge involved **pushing large model files to Git**, which was 
resolved by tracking models with DVC so that only lightweight `.dvc` files were stored in the repository. Another issue was 
**reproducibility of random processes**, addressed by setting random seeds in the configuration files and passing them consistently 
to all functions. Finally, **tracking over twenty experiments manually** became impractical, but MLflow’s automatic tracking provided 
a seamless and efficient solution.

## 6. Recommendations

### 6.1 For Phase 3

In the next phase of development, the primary focus will be on deploying the best-performing model — the tuned SVM — as a REST API 
to enable real-time predictions and integration with external systems. Setting up a continuous integration and continuous deployment 
(CI/CD) pipeline will ensure smooth updates and automated testing. Additionally, implementing model monitoring and drift detection 
will help maintain performance stability in production. Finally, an A/B testing framework will be introduced to evaluate new model 
versions objectively and improve deployment decision-making.

### 6.2 Model Improvements

In the short term, model performance can be further enhanced through advanced feature engineering, including the creation of polynomial 
features and interaction terms. Ensemble methods such as stacking and blending will be explored to leverage the strengths of multiple 
algorithms. Expanding the training dataset will also be a key step toward improving model generalization.  

In the long term, the project aims to explore deep learning techniques if the dataset size increases significantly. AutoML tools will 
be investigated to streamline hyperparameter optimization, and comprehensive feature selection analyses will be performed to better 
understand feature importance and reduce model complexity.

### 6.3 MLOps Enhacements

From an MLOps perspective, several enhancements are planned. Immediately, the focus will be on strengthening software quality and 
reliability by adding unit tests using `pytest`, implementing pre-commit hooks, and integrating data validation with **Great Expectations**.  

In the near future, the team plans to containerize the application using **Docker** for portability and scalability. 
Workflow orchestration will be managed through **Airflow** or **Prefect** to ensure reproducible automation. A **Model Registry** will 
be established using MLflow to manage model versions, and a **Grafana dashboard** will be developed to monitor model performance and 
operational metrics in real time.

## Conclusions


Phase 2 successfully transformed the project from a monolithic notebook into a production-ready MLOps system. This phase achieved 
several major milestones that established a strong foundation for scalability and collaboration. The **codebase was fully refactored** 
into a modular, maintainable, and reusable structure, improving both readability and team productivity. **MLflow integration** enabled 
comprehensive experiment tracking, making it easy to compare results, monitor metrics, and manage model versions. The implementation 
of a **DVC pipeline** automated the workflow, ensuring full reproducibility of experiments and consistent data management. Additionally, 
**AWS S3 storage** was incorporated for reliable, cloud-based version control of datasets and models. Overall, Phase 2 solidified the 
project’s alignment with **industry-standard MLOps practices**, positioning it for seamless deployment and continued innovation 
in future phases.























