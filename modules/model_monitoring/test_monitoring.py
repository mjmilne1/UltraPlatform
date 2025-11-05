import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("🔍 Testing Model Monitoring System")

# Test drift detection with KS test
def test_drift_detection():
    # Create two distributions
    baseline = np.random.normal(0, 1, 1000)
    current_with_drift = np.random.normal(0.5, 1, 1000)  # Shifted mean
    current_no_drift = np.random.normal(0, 1, 1000)
    
    # Test with drift
    statistic_drift, p_value_drift = ks_2samp(baseline, current_with_drift)
    print(f"\n📊 Drift Detection Test:")
    print(f"With drift - KS statistic: {statistic_drift:.3f}, p-value: {p_value_drift:.3f}")
    print(f"Drift detected: {'Yes' if p_value_drift < 0.05 else 'No'}")
    
    # Test without drift
    statistic_no_drift, p_value_no_drift = ks_2samp(baseline, current_no_drift)
    print(f"No drift - KS statistic: {statistic_no_drift:.3f}, p-value: {p_value_no_drift:.3f}")
    print(f"Drift detected: {'Yes' if p_value_no_drift < 0.05 else 'No'}")

# Test performance monitoring
def test_performance_monitoring():
    # Simulate predictions and actuals
    predictions = np.random.randint(0, 2, 100)
    actuals = np.random.randint(0, 2, 100)
    
    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, zero_division=0)
    recall = recall_score(actuals, predictions, zero_division=0)
    f1 = f1_score(actuals, predictions, zero_division=0)
    
    print(f"\n📈 Performance Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Determine model health
    if accuracy < 0.5:
        status = "FAILING"
    elif accuracy < 0.7:
        status = "DEGRADED"
    elif accuracy < 0.8:
        status = "WARNING"
    else:
        status = "HEALTHY"
    
    print(f"Model Status: {status}")

# Test Wasserstein distance for distribution comparison
def test_wasserstein_distance():
    dist1 = np.random.normal(0, 1, 1000)
    dist2 = np.random.normal(0.3, 1.1, 1000)
    
    w_distance = wasserstein_distance(dist1, dist2)
    print(f"\n📏 Wasserstein Distance Test:")
    print(f"Distance between distributions: {w_distance:.3f}")
    print(f"Significant difference: {'Yes' if w_distance > 0.2 else 'No'}")

if __name__ == '__main__':
    print("="*60)
    print("MODEL MONITORING SYSTEM - FUNCTIONALITY TEST")
    print("="*60)
    
    test_drift_detection()
    test_performance_monitoring()
    test_wasserstein_distance()
    
    print("\n✅ All tests completed successfully!")
