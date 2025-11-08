""" 
Data Drift Detection using Evidently
Monitors data distribution changes over time
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DataDriftMetric, ColumnDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning('Evidenyly not available. Drift detection disabled.')
    
from app.config import config

logger = logging.getLogger(__name__)

class DriftDetector:
    """Class to handle data drift detection."""
    
    def __init__(self, reference_data_path: Optional[Path] = None):
        """
        Initialize drift detector. 
        Args:
            reference_data path: Path to reference data CSV 
        """
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.drift_threshold = 0.1 # 10% drift threshold
        
        if not EVIDENTLY_AVAILABLE:
            logger.warning('Evidently not installed. Drift detection unavailable.')
            return
        
        # Load reference data if available
        if reference_data_path and reference_data_path.exists():
            self.load_reference_data(reference_data_path)
        else:
            logger.warning(f'Reference data not found at {reference_data_path}')
            
    def load_reference_data(self, path: Path):
        """Load reference data from CSV"""
        try:
            self.reference_data = pd.read_csv(path)
            logger.info(f"Loaded reference data: {self.reference_data.shape}")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            self.reference_data = None
            
    def save_reference_data(self, data: pd.DataFrame, path: Path):
        """Save reference data to CSV"""
        try: 
            path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, index=False)
            self.reference_data = data
            logger.info(f"Saved reference data to {path}")
        except Exception as e:
            logger.error(f"Failed to save reference data: {e}")
            
    def detect_drift(
        self, 
        current_data: pd.DataFrame, 
        reference_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """ 
        Detect drift between reference and current data. 
        
        Args:
            current_data: Current data DataFrame
            reference_data: Reference data (if None, uses stored reference)
        Returns:
            Dictionary with drift detection results
        """
        
        if not EVIDENTLY_AVAILABLE:
            return {
                'error': 'Evidently not installed', 
                'drift_detected': False, 
                'timestamp': datetime.now().isoformat()
            }
        
        # Use stored reference if not provided
        if reference_data is None:
            reference_data = self.reference_data
            
        if reference_data is None:
            return {
                'error': 'No reference data available', 
                'drift_detected': False, 
                'timestamp': datetime.now().isoformat()
            }
            
        try:
            # Ensure both datasets have same columns
            common_cols = list(set(reference_data.columns) & set(current_data.columns))
            ref_data = reference_data[common_cols]
            curr_data = current_data[common_cols]
            
            # Create drift report
            report = Report(metrics=[
                DatasetDriftMetric(),
            ])
            
            report.run(
                reference_data=ref_data, 
                current_data=curr_data
            )
            
            # Extract results
            results = report.as_dict()
            
            # Parse drift metrics
            drift_metrics = self._parse_drift_results(results)
            
            logger.info(f"Drift detection complete: {drift_metrics['dataset_drift']}")
            
            return drift_metrics
        
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {
                'error': str(e), 
                'drift_detected': False, 
                'timestamp': datetime.now().isoformat()
            }
            
    def _parse_drift_results(self, results: Dict) -> Dict[str, Any]:
        """Parse Evidently drift results into simplified format"""
        try:
            metrics = results.get('metrics', [{}])[0]
            result_data = metrics.get('result', {})
            
            dataset_drift = result_data.get('dataset_drift', False)
            drift_share = result_data.get('drift_share', 0.0)
            number_of_drifted_columns = result_data.get('number_of_drifted_columns', 0)
            
            return {
                'drift_detected': dataset_drift, 
                'drift_share': drift_share, 
                'drifted_columns_count': number_of_drifted_columns, 
                'drift_score': drift_share, 
                'threshold': self.drift_threshold, 
                'timestamp': datetime.now().isoformat(), 
                'reference_size': len(self.reference_data) if self.reference_data is not None else 0
            }
        except Exception as e:
            logger.error(f"Failed to parse drift results: {e}")
            return {
                'drift_detected': False,
                'error': f'Failed to parse results: {str(e)}', 
                'timestamp': datetime.now().isoformat()
            }
            
    def get_column_drift(
        self,
        current_data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        reference_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Get drift metrics for specific columns.
        
        Args:
            current_data: Current data
            columns: List of columns to check (if None, checks all)
            reference_data: Reference data (if None, uses stored)
            
        Returns:
            Dictionary with per-column drift metrics
        """
        if not EVIDENTLY_AVAILABLE:
            return {"error": "Evidently not installed"}
        
        if reference_data is None:
            reference_data = self.reference_data
        
        if reference_data is None:
            return {"error": "No reference data available"}
        
        try:
            if columns is None:
                columns = list(current_data.columns)
            
            column_metrics = []
            
            for col in columns:
                if col not in current_data.columns or col not in reference_data.columns:
                    continue
                
                report = Report(metrics=[
                    ColumnDriftMetric(column_name=col)
                ])
                
                report.run(
                    reference_data=reference_data[[col]],
                    current_data=current_data[[col]]
                )
                
                results = report.as_dict()
                drift_detected = results['metrics'][0]['result'].get('drift_detected', False)
                drift_score = results['metrics'][0]['result'].get('drift_score', 0.0)
                
                column_metrics.append({
                    "column": col,
                    "drift_detected": drift_detected,
                    "drift_score": drift_score
                })
            
            return {
                "columns": column_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Column drift detection failed: {e}")
            return {"error": str(e)}


# Global drift detector instance
_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get or create global drift detector instance."""
    global _drift_detector
    
    if _drift_detector is None:
        # Try to load reference data
        ref_data_path = config.PROJECT_ROOT / "data" / "reference" / "reference_data.csv"
        _drift_detector = DriftDetector(reference_data_path=ref_data_path)
    
    return _drift_detector