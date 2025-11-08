""" 
Quick script to train a simple model for API testing 
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# Create synthetic training data
np.random.seed(42)
n_samples = 200

X = pd.DataFrame({
    'Class_X_Percentage': np.random.uniform(50, 95, n_samples),
    'Class_XII_Percentage': np.random.uniform(50, 95, n_samples),
    'Study_Hours': np.random.uniform(1, 10, n_samples),
    'Gender_Male': np.random.choice([0, 1], n_samples),
    'Caste_General': np.random.choice([0, 1], n_samples),
    'Caste_OBC': np.random.choice([0, 1], n_samples),
    'Caste_SC': np.random.choice([0, 1], n_samples),
    'Coaching_Yes': np.random.choice([0, 1], n_samples),
    'Medium_English': np.random.choice([0, 1], n_samples)
})

# Create target (high performance if good grades + study hours)
y = ((X['Class_X_Percentage'] > 70) & 
     (X['Class_XII_Percentage'] > 70) & 
     (X['Study_Hours'] > 4)).astype(int)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train
print("Training model...")
pipeline.fit(X, y)

# Save
model_path = Path('models/pipeline_baseline.pkl')
model_path.parent.mkdir(exist_ok=True)
joblib.dump(pipeline, model_path)

print(f"Model saved to: {model_path}")
print(f"Model type: {type(pipeline).__name__}")
print(f"Features: {X.columns.tolist()}")
print(f"Classes: {pipeline.classes_}")