from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
import hashlib
from pathlib import Path

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# ML imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

print("🔍 Model Monitoring & Drift Detection Module Loaded")

# ==================== ENUMS ====================

class DriftType(Enum):
    DATA_DRIFT = 'data_drift'
    CONCEPT_DRIFT = 'concept_drift'
    PREDICTION_DRIFT = 'prediction_drift'
    PERFORMANCE_DRIFT = 'performance_drift'
    FEATURE_DRIFT = 'feature_drift'
    LABEL_DRIFT = 'label_drift'

class DriftSeverity(Enum):
    NONE = 'none'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class MonitoringMetric(Enum):
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'
    AUC_ROC = 'auc_roc'
    MSE = 'mse'
    MAE = 'mae'
    RMSE = 'rmse'
    R2 = 'r2'
    LATENCY = 'latency'
    THROUGHPUT = 'throughput'

class AlertLevel(Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

class ModelStatus(Enum):
    HEALTHY = 'healthy'
    WARNING = 'warning'
    DEGRADED = 'degraded'
    FAILING = 'failing'
    OFFLINE = 'offline'

# ==================== DATA CLASSES ====================

@dataclass
class DriftReport:
    '''Drift detection report'''
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    model_id: str = ''
    drift_type: DriftType = DriftType.DATA_DRIFT
    severity: DriftSeverity = DriftSeverity.NONE
    drift_score: float = 0.0
    p_value: float = 1.0
    features_affected: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    '''Model performance report'''
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    model_id: str = ''
    period: str = 'daily'
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    degradation: Dict[str, float] = field(default_factory=dict)
    status: ModelStatus = ModelStatus.HEALTHY
    alerts: List[Dict] = field(default_factory=list)

@dataclass
class MonitoringAlert:
    '''Monitoring alert'''
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    model_id: str = ''
    alert_level: AlertLevel = AlertLevel.INFO
    alert_type: str = ''
    message: str = ''
    metrics: Dict[str, Any] = field(default_factory=dict)
    action_required: str = ''
    auto_resolved: bool = False

@dataclass
class ModelMonitoringConfig:
    '''Configuration for model monitoring'''
    model_id: str
    model_name: str
    monitoring_frequency: str = 'hourly'  # hourly, daily, weekly
    drift_threshold: float = 0.05  # p-value threshold
    performance_threshold: float = 0.05  # 5% degradation threshold
    alert_enabled: bool = True
    auto_retrain: bool = False
    baseline_window: int = 30  # days
    monitoring_window: int = 7  # days

# ==================== MAIN MONITORING SYSTEM ====================

class ModelMonitoringSystem:
    '''Comprehensive Model Monitoring & Drift Detection System'''
    
    def __init__(self, storage_path: str = './monitoring_data'):
        self.name = 'UltraPlatform Model Monitoring System'
        self.version = '2.0'
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.drift_detector = DriftDetector()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self.visualization_engine = VisualizationEngine()
        self.report_generator = ReportGenerator()
        
        # Storage
        self.monitoring_data = defaultdict(lambda: {
            'predictions': deque(maxlen=10000),
            'actuals': deque(maxlen=10000),
            'features': deque(maxlen=10000),
            'timestamps': deque(maxlen=10000),
            'metrics': deque(maxlen=1000),
            'alerts': deque(maxlen=100)
        })
        
        # Baseline data
        self.baseline_data = {}
        
        # Model configurations
        self.model_configs = {}
        
        print('✅ Model Monitoring System initialized')
    
    def register_model(self, config: ModelMonitoringConfig, baseline_data: pd.DataFrame = None):
        '''Register a model for monitoring'''
        print(f'\n📝 Registering model: {config.model_name}')
        
        self.model_configs[config.model_id] = config
        
        if baseline_data is not None:
            self.baseline_data[config.model_id] = {
                'features': baseline_data,
                'statistics': self._calculate_baseline_statistics(baseline_data),
                'timestamp': datetime.now()
            }
            print(f'   ✅ Baseline established with {len(baseline_data)} samples')
        
        return True
    
    def monitor_prediction(self, 
                          model_id: str,
                          features: Union[Dict, pd.DataFrame],
                          prediction: Any,
                          actual: Any = None,
                          timestamp: datetime = None):
        '''Monitor a single prediction'''
        
        if model_id not in self.model_configs:
            print(f'⚠️ Model {model_id} not registered')
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store monitoring data
        self.monitoring_data[model_id]['predictions'].append(prediction)
        self.monitoring_data[model_id]['features'].append(features)
        self.monitoring_data[model_id]['timestamps'].append(timestamp)
        
        if actual is not None:
            self.monitoring_data[model_id]['actuals'].append(actual)
        
        # Check if we should run monitoring
        if self._should_run_monitoring(model_id):
            self.run_monitoring_check(model_id)
    
    def run_monitoring_check(self, model_id: str):
        '''Run comprehensive monitoring check'''
        print(f'\n🔍 MONITORING CHECK - Model: {model_id}')
        print('='*60)
        
        config = self.model_configs[model_id]
        
        # Step 1: Check for Data Drift
        print('\n1️⃣ DATA DRIFT DETECTION')
        print('-'*40)
        
        if model_id in self.baseline_data:
            data_drift = self.drift_detector.detect_data_drift(
                self.baseline_data[model_id]['features'],
                self._get_recent_features(model_id),
                threshold=config.drift_threshold
            )
            
            print(f'   Drift Detected: {"Yes" if data_drift.severity != DriftSeverity.NONE else "No"}')
            print(f'   Severity: {data_drift.severity.value}')
            print(f'   Drift Score: {data_drift.drift_score:.3f}')
            print(f'   P-value: {data_drift.p_value:.3f}')
            
            if data_drift.features_affected:
                print(f'   Affected Features: {", ".join(data_drift.features_affected[:5])}')
        
        # Step 2: Check Performance
        print('\n2️⃣ PERFORMANCE MONITORING')
        print('-'*40)
        
        performance = self.performance_monitor.check_performance(
            self._get_recent_predictions(model_id),
            self._get_recent_actuals(model_id),
            model_type='classification'
        )
        
        print(f'   Current Accuracy: {performance.metrics.get("accuracy", 0):.3f}')
        print(f'   Current F1-Score: {performance.metrics.get("f1_score", 0):.3f}')
        print(f'   Status: {performance.status.value}')
        
        # Compare with baseline
        if performance.baseline_metrics:
            for metric, value in performance.metrics.items():
                baseline = performance.baseline_metrics.get(metric, value)
                change = ((value - baseline) / baseline * 100) if baseline != 0 else 0
                if abs(change) > 5:
                    print(f'   ⚠️ {metric}: {change:+.1f}% change from baseline')
        
        # Step 3: Prediction Drift
        print('\n3️⃣ PREDICTION DRIFT DETECTION')
        print('-'*40)
        
        pred_drift = self.drift_detector.detect_prediction_drift(
            self._get_historical_predictions(model_id),
            self._get_recent_predictions(model_id)
        )
        
        print(f'   Drift Detected: {"Yes" if pred_drift.severity != DriftSeverity.NONE else "No"}')
        print(f'   Severity: {pred_drift.severity.value}')
        print(f'   Distribution Shift: {pred_drift.drift_score:.3f}')
        
        # Step 4: Concept Drift
        print('\n4️⃣ CONCEPT DRIFT DETECTION')
        print('-'*40)
        
        if len(self.monitoring_data[model_id]['actuals']) > 100:
            concept_drift = self.drift_detector.detect_concept_drift(
                self._get_recent_features(model_id),
                self._get_recent_predictions(model_id),
                self._get_recent_actuals(model_id)
            )
            
            print(f'   Drift Detected: {"Yes" if concept_drift.severity != DriftSeverity.NONE else "No"}')
            print(f'   Severity: {concept_drift.severity.value}')
            print(f'   Relationship Change: {concept_drift.drift_score:.3f}')
        else:
            print('   Insufficient data for concept drift detection')
        
        # Step 5: Generate Alerts
        print('\n5️⃣ ALERT GENERATION')
        print('-'*40)
        
        alerts = []
        
        # Data drift alert
        if data_drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            alert = self.alert_manager.create_alert(
                model_id=model_id,
                alert_level=AlertLevel.WARNING if data_drift.severity == DriftSeverity.HIGH else AlertLevel.CRITICAL,
                alert_type='data_drift',
                message=f'Significant data drift detected: {data_drift.severity.value}',
                metrics={'drift_score': data_drift.drift_score, 'p_value': data_drift.p_value}
            )
            alerts.append(alert)
            print(f'   ⚠️ {alert.alert_level.value.upper()}: {alert.message}')
        
        # Performance alert
        if performance.status in [ModelStatus.DEGRADED, ModelStatus.FAILING]:
            alert = self.alert_manager.create_alert(
                model_id=model_id,
                alert_level=AlertLevel.ERROR if performance.status == ModelStatus.FAILING else AlertLevel.WARNING,
                alert_type='performance_degradation',
                message=f'Model performance degraded: {performance.status.value}',
                metrics=performance.degradation
            )
            alerts.append(alert)
            print(f'   ⚠️ {alert.alert_level.value.upper()}: {alert.message}')
        
        if not alerts:
            print('   ✅ No alerts generated')
        
        # Step 6: Recommendations
        print('\n6️⃣ RECOMMENDATIONS')
        print('-'*40)
        
        recommendations = self._generate_recommendations(
            data_drift, performance, pred_drift, config
        )
        
        for rec in recommendations:
            print(f'   • {rec}')
        
        # Store monitoring results
        self._store_monitoring_results(model_id, {
            'data_drift': data_drift,
            'performance': performance,
            'prediction_drift': pred_drift,
            'alerts': alerts,
            'timestamp': datetime.now()
        })
        
        return {
            'data_drift': data_drift,
            'performance': performance,
            'prediction_drift': pred_drift,
            'alerts': alerts,
            'recommendations': recommendations
        }
    
    def generate_monitoring_report(self, model_id: str, period: str = 'daily'):
        '''Generate comprehensive monitoring report'''
        print(f'\n📊 GENERATING MONITORING REPORT')
        print('='*60)
        
        report = self.report_generator.generate_report(
            model_id,
            self.monitoring_data[model_id],
            self.model_configs.get(model_id),
            period
        )
        
        # Create visualizations
        self.visualization_engine.create_monitoring_dashboard(
            model_id,
            self.monitoring_data[model_id],
            self.storage_path / f'dashboard_{model_id}.html'
        )
        
        print(f'✅ Report generated: {report["summary"]}')
        
        return report
    
    def _should_run_monitoring(self, model_id: str) -> bool:
        '''Check if monitoring should run'''
        # Simple check - run every 100 predictions
        return len(self.monitoring_data[model_id]['predictions']) % 100 == 0
    
    def _calculate_baseline_statistics(self, data: pd.DataFrame):
        '''Calculate baseline statistics'''
        stats = {}
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                stats[column] = {
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std()),
                    'min': float(data[column].min()),
                    'max': float(data[column].max()),
                    'q25': float(data[column].quantile(0.25)),
                    'median': float(data[column].median()),
                    'q75': float(data[column].quantile(0.75))
                }
            else:
                # Categorical
                value_counts = data[column].value_counts()
                stats[column] = {
                    'unique_values': len(value_counts),
                    'distribution': value_counts.to_dict()
                }
        
        return stats
    
    def _get_recent_features(self, model_id: str) -> pd.DataFrame:
        '''Get recent feature data'''
        features = list(self.monitoring_data[model_id]['features'])[-1000:]
        if features and isinstance(features[0], dict):
            return pd.DataFrame(features)
        return pd.DataFrame()
    
    def _get_recent_predictions(self, model_id: str) -> np.ndarray:
        '''Get recent predictions'''
        return np.array(list(self.monitoring_data[model_id]['predictions'])[-1000:])
    
    def _get_recent_actuals(self, model_id: str) -> np.ndarray:
        '''Get recent actuals'''
        return np.array(list(self.monitoring_data[model_id]['actuals'])[-1000:])
    
    def _get_historical_predictions(self, model_id: str) -> np.ndarray:
        '''Get historical predictions'''
        preds = list(self.monitoring_data[model_id]['predictions'])
        if len(preds) > 2000:
            return np.array(preds[-2000:-1000])
        return np.array(preds[:len(preds)//2])
    
    def _generate_recommendations(self, data_drift, performance, pred_drift, config):
        '''Generate actionable recommendations'''
        recommendations = []
        
        # Data drift recommendations
        if data_drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.append('Retrain model with recent data to address data drift')
            recommendations.append('Review and update feature engineering pipeline')
        elif data_drift.severity == DriftSeverity.MEDIUM:
            recommendations.append('Monitor data drift closely - consider retraining soon')
        
        # Performance recommendations
        if performance.status == ModelStatus.FAILING:
            recommendations.append('URGENT: Model failing - immediate retraining required')
            recommendations.append('Consider fallback to previous model version')
        elif performance.status == ModelStatus.DEGRADED:
            recommendations.append('Schedule model retraining to improve performance')
        
        # Prediction drift recommendations
        if pred_drift.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.append('Investigate changes in prediction distribution')
            recommendations.append('Check for upstream data pipeline issues')
        
        # General recommendations
        if config.auto_retrain and (data_drift.severity != DriftSeverity.NONE or 
                                   performance.status != ModelStatus.HEALTHY):
            recommendations.append('Auto-retraining triggered based on monitoring results')
        
        if not recommendations:
            recommendations.append('Model performing within expected parameters')
        
        return recommendations
    
    def _store_monitoring_results(self, model_id: str, results: Dict):
        '''Store monitoring results'''
        # Save to file
        results_file = self.storage_path / f'monitoring_{model_id}_{datetime.now():%Y%m%d_%H%M%S}.json'
        
        # Convert to serializable format
        serializable_results = {
            'model_id': model_id,
            'timestamp': results['timestamp'].isoformat(),
            'data_drift': {
                'severity': results['data_drift'].severity.value,
                'drift_score': results['data_drift'].drift_score,
                'p_value': results['data_drift'].p_value
            },
            'performance': {
                'status': results['performance'].status.value,
                'metrics': results['performance'].metrics
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

# ==================== DRIFT DETECTION ====================

class DriftDetector:
    '''Drift detection component'''
    
    def detect_data_drift(self, 
                         baseline_data: pd.DataFrame, 
                         current_data: pd.DataFrame,
                         threshold: float = 0.05) -> DriftReport:
        '''Detect data drift using statistical tests'''
        
        if baseline_data.empty or current_data.empty:
            return DriftReport(drift_type=DriftType.DATA_DRIFT)
        
        drift_scores = []
        p_values = []
        features_affected = []
        
        # Ensure same columns
        common_cols = set(baseline_data.columns) & set(current_data.columns)
        
        for column in common_cols:
            baseline_col = baseline_data[column]
            current_col = current_data[column]
            
            # Skip if not enough data
            if len(baseline_col) < 10 or len(current_col) < 10:
                continue
            
            # Numerical features - KS test
            if baseline_col.dtype in ['int64', 'float64']:
                statistic, p_value = ks_2samp(baseline_col, current_col)
                
                if p_value < threshold:
                    features_affected.append(column)
                    drift_scores.append(statistic)
                    p_values.append(p_value)
            
            # Categorical features - Chi-square test
            else:
                try:
                    # Create frequency table
                    baseline_freq = baseline_col.value_counts()
                    current_freq = current_col.value_counts()
                    
                    # Align categories
                    all_categories = set(baseline_freq.index) | set(current_freq.index)
                    baseline_aligned = [baseline_freq.get(cat, 0) for cat in all_categories]
                    current_aligned = [current_freq.get(cat, 0) for cat in all_categories]
                    
                    # Chi-square test
                    chi2, p_value = stats.chisquare(current_aligned, baseline_aligned)
                    
                    if p_value < threshold:
                        features_affected.append(column)
                        drift_scores.append(chi2)
                        p_values.append(p_value)
                except:
                    pass
        
        # Aggregate drift score
        if drift_scores:
            avg_drift_score = np.mean(drift_scores)
            min_p_value = min(p_values)
        else:
            avg_drift_score = 0
            min_p_value = 1.0
        
        # Determine severity
        severity = self._determine_drift_severity(avg_drift_score, min_p_value, len(features_affected), len(common_cols))
        
        # Generate recommendations
        recommendations = []
        if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.append('Immediate model retraining recommended')
        elif severity == DriftSeverity.MEDIUM:
            recommendations.append('Monitor closely and schedule retraining')
        
        return DriftReport(
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            drift_score=avg_drift_score,
            p_value=min_p_value,
            features_affected=features_affected,
            recommendations=recommendations,
            details={
                'total_features': len(common_cols),
                'affected_features': len(features_affected),
                'drift_percentage': len(features_affected) / len(common_cols) * 100 if common_cols else 0
            }
        )
    
    def detect_prediction_drift(self, 
                               historical_predictions: np.ndarray,
                               current_predictions: np.ndarray) -> DriftReport:
        '''Detect drift in prediction distribution'''
        
        if len(historical_predictions) == 0 or len(current_predictions) == 0:
            return DriftReport(drift_type=DriftType.PREDICTION_DRIFT)
        
        # KS test for prediction distribution
        statistic, p_value = ks_2samp(historical_predictions, current_predictions)
        
        # Wasserstein distance
        w_distance = wasserstein_distance(historical_predictions, current_predictions)
        
        # Check for mean shift
        hist_mean = np.mean(historical_predictions)
        curr_mean = np.mean(current_predictions)
        mean_shift = abs(curr_mean - hist_mean) / (hist_mean + 1e-10)
        
        # Severity based on multiple factors
        severity = self._determine_drift_severity(statistic, p_value, mean_shift, 1)
        
        return DriftReport(
            drift_type=DriftType.PREDICTION_DRIFT,
            severity=severity,
            drift_score=statistic,
            p_value=p_value,
            recommendations=['Investigate prediction distribution changes'] if severity != DriftSeverity.NONE else [],
            details={
                'wasserstein_distance': w_distance,
                'mean_shift': mean_shift,
                'historical_mean': hist_mean,
                'current_mean': curr_mean
            }
        )
    
    def detect_concept_drift(self,
                           features: pd.DataFrame,
                           predictions: np.ndarray,
                           actuals: np.ndarray) -> DriftReport:
        '''Detect concept drift (relationship change)'''
        
        if len(predictions) != len(actuals) or len(predictions) < 50:
            return DriftReport(drift_type=DriftType.CONCEPT_DRIFT)
        
        # Split data into windows
        mid_point = len(predictions) // 2
        
        # Calculate accuracy in each window
        first_half_acc = accuracy_score(actuals[:mid_point], predictions[:mid_point])
        second_half_acc = accuracy_score(actuals[mid_point:], predictions[mid_point:])
        
        # Accuracy degradation
        acc_change = abs(first_half_acc - second_half_acc)
        
        # Statistical test for proportion difference
        n1, n2 = mid_point, len(predictions) - mid_point
        p1, p2 = first_half_acc, second_half_acc
        
        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        
        # Z-statistic
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z_stat = abs(p1 - p2) / (se + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Determine severity
        severity = DriftSeverity.NONE
        if acc_change > 0.15:
            severity = DriftSeverity.CRITICAL
        elif acc_change > 0.10:
            severity = DriftSeverity.HIGH
        elif acc_change > 0.05:
            severity = DriftSeverity.MEDIUM
        elif acc_change > 0.02:
            severity = DriftSeverity.LOW
        
        return DriftReport(
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=severity,
            drift_score=acc_change,
            p_value=p_value,
            recommendations=['Model retraining required'] if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] else [],
            details={
                'first_window_accuracy': first_half_acc,
                'second_window_accuracy': second_half_acc,
                'accuracy_change': acc_change,
                'z_statistic': z_stat
            }
        )
    
    def _determine_drift_severity(self, drift_score: float, p_value: float, 
                                 affected_ratio: float, total: float) -> DriftSeverity:
        '''Determine drift severity'''
        
        # P-value based severity
        if p_value < 0.001:
            p_severity = 4
        elif p_value < 0.01:
            p_severity = 3
        elif p_value < 0.05:
            p_severity = 2
        elif p_value < 0.1:
            p_severity = 1
        else:
            p_severity = 0
        
        # Affected features ratio
        if total > 0:
            ratio = affected_ratio / total
            if ratio > 0.5:
                ratio_severity = 4
            elif ratio > 0.3:
                ratio_severity = 3
            elif ratio > 0.15:
                ratio_severity = 2
            elif ratio > 0.05:
                ratio_severity = 1
            else:
                ratio_severity = 0
        else:
            ratio_severity = 0
        
        # Combined severity
        combined = max(p_severity, ratio_severity)
        
        if combined >= 4:
            return DriftSeverity.CRITICAL
        elif combined == 3:
            return DriftSeverity.HIGH
        elif combined == 2:
            return DriftSeverity.MEDIUM
        elif combined == 1:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

# ==================== PERFORMANCE MONITORING ====================

class PerformanceMonitor:
    '''Performance monitoring component'''
    
    def check_performance(self, predictions: np.ndarray, actuals: np.ndarray, 
                         model_type: str = 'classification') -> PerformanceReport:
        '''Check model performance'''
        
        if len(predictions) == 0 or len(actuals) == 0 or len(predictions) != len(actuals):
            return PerformanceReport(status=ModelStatus.OFFLINE)
        
        metrics = {}
        
        if model_type == 'classification':
            metrics['accuracy'] = accuracy_score(actuals, predictions)
            metrics['precision'] = precision_score(actuals, predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(actuals, predictions, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(actuals, predictions, average='weighted', zero_division=0)
        else:
            metrics['mse'] = mean_squared_error(actuals, predictions)
            metrics['mae'] = mean_absolute_error(actuals, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(actuals, predictions)
        
        # Determine status based on performance
        status = self._determine_model_status(metrics, model_type)
        
        return PerformanceReport(
            metrics=metrics,
            status=status,
            period='current'
        )
    
    def _determine_model_status(self, metrics: Dict, model_type: str) -> ModelStatus:
        '''Determine model status from metrics'''
        
        if model_type == 'classification':
            accuracy = metrics.get('accuracy', 0)
            
            if accuracy < 0.5:
                return ModelStatus.FAILING
            elif accuracy < 0.7:
                return ModelStatus.DEGRADED
            elif accuracy < 0.8:
                return ModelStatus.WARNING
            else:
                return ModelStatus.HEALTHY
        else:
            r2 = metrics.get('r2', 0)
            
            if r2 < 0.3:
                return ModelStatus.FAILING
            elif r2 < 0.5:
                return ModelStatus.DEGRADED
            elif r2 < 0.7:
                return ModelStatus.WARNING
            else:
                return ModelStatus.HEALTHY

# ==================== ALERT MANAGEMENT ====================

class AlertManager:
    '''Alert management component'''
    
    def __init__(self):
        self.alerts = deque(maxlen=1000)
    
    def create_alert(self, model_id: str, alert_level: AlertLevel, 
                    alert_type: str, message: str, metrics: Dict = None) -> MonitoringAlert:
        '''Create a monitoring alert'''
        
        action_required = self._determine_action(alert_level, alert_type)
        
        alert = MonitoringAlert(
            model_id=model_id,
            alert_level=alert_level,
            alert_type=alert_type,
            message=message,
            metrics=metrics or {},
            action_required=action_required
        )
        
        self.alerts.append(alert)
        
        # Send alert (in production would integrate with notification system)
        self._send_alert(alert)
        
        return alert
    
    def _determine_action(self, level: AlertLevel, alert_type: str) -> str:
        '''Determine required action'''
        
        if level == AlertLevel.CRITICAL:
            return 'Immediate investigation and remediation required'
        elif level == AlertLevel.ERROR:
            return 'Investigation required within 24 hours'
        elif level == AlertLevel.WARNING:
            return 'Monitor closely, schedule review'
        else:
            return 'No immediate action required'
    
    def _send_alert(self, alert: MonitoringAlert):
        '''Send alert (placeholder for notification system)'''
        print(f'🚨 ALERT: [{alert.alert_level.value.upper()}] {alert.message}')

# ==================== VISUALIZATION ====================

class VisualizationEngine:
    '''Visualization component'''
    
    def create_monitoring_dashboard(self, model_id: str, data: Dict, output_path: Path):
        '''Create monitoring dashboard'''
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Model Monitoring Dashboard - {model_id}', fontsize=16)
        
        # 1. Prediction Distribution Over Time
        if data['predictions']:
            predictions = list(data['predictions'])
            axes[0, 0].hist(predictions[-1000:], bins=30, alpha=0.7)
            axes[0, 0].set_title('Recent Prediction Distribution')
            axes[0, 0].set_xlabel('Prediction Value')
            axes[0, 0].set_ylabel('Frequency')
        
        # 2. Performance Metrics Over Time
        if data['metrics']:
            metrics_df = pd.DataFrame(list(data['metrics'])[-100:])
            if not metrics_df.empty and 'accuracy' in metrics_df.columns:
                axes[0, 1].plot(metrics_df['accuracy'])
                axes[0, 1].set_title('Accuracy Over Time')
                axes[0, 1].set_xlabel('Time')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].set_ylim([0, 1])
        
        # 3. Alert Frequency
        if data['alerts']:
            alert_levels = [alert.alert_level.value for alert in list(data['alerts'])[-50:]]
            alert_counts = pd.Series(alert_levels).value_counts()
            axes[0, 2].bar(alert_counts.index, alert_counts.values)
            axes[0, 2].set_title('Recent Alerts by Level')
            axes[0, 2].set_xlabel('Alert Level')
            axes[0, 2].set_ylabel('Count')
        
        # 4. Actual vs Predicted
        if data['predictions'] and data['actuals']:
            preds = list(data['predictions'])[-100:]
            acts = list(data['actuals'])[-100:]
            if len(preds) == len(acts):
                axes[1, 0].scatter(acts, preds, alpha=0.5)
                axes[1, 0].plot([min(acts), max(acts)], [min(acts), max(acts)], 'r--')
                axes[1, 0].set_title('Actual vs Predicted')
                axes[1, 0].set_xlabel('Actual')
                axes[1, 0].set_ylabel('Predicted')
        
        # 5. Drift Score Timeline
        axes[1, 1].set_title('Drift Detection Timeline')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Drift Score')
        
        # 6. Model Status
        status_text = 'Model Status: HEALTHY'
        axes[1, 2].text(0.5, 0.5, status_text, ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle='round', facecolor='lightgreen'))
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = output_path.with_suffix('.png')
        plt.savefig(dashboard_path)
        plt.close()
        
        print(f'   📊 Dashboard saved to: {dashboard_path}')
        
        return str(dashboard_path)

# ==================== REPORT GENERATION ====================

class ReportGenerator:
    '''Report generation component'''
    
    def generate_report(self, model_id: str, data: Dict, config: ModelMonitoringConfig, period: str) -> Dict:
        '''Generate monitoring report'''
        
        report = {
            'model_id': model_id,
            'model_name': config.model_name if config else 'Unknown',
            'period': period,
            'timestamp': datetime.now().isoformat(),
            'summary': 'Model monitoring report generated successfully',
            'sections': {}
        }
        
        # Performance section
        if data['predictions'] and data['actuals']:
            predictions = np.array(list(data['predictions'])[-1000:])
            actuals = np.array(list(data['actuals'])[-1000:])
            
            if len(predictions) == len(actuals):
                report['sections']['performance'] = {
                    'accuracy': float(accuracy_score(actuals, predictions)),
                    'sample_count': len(predictions),
                    'period': f'Last {len(predictions)} predictions'
                }
        
        # Drift section
        report['sections']['drift'] = {
            'checks_performed': ['data_drift', 'prediction_drift', 'concept_drift'],
            'alerts_raised': len([a for a in data['alerts'] if hasattr(a, 'alert_level')])
        }
        
        # Recommendations
        report['sections']['recommendations'] = [
            'Continue monitoring model performance',
            'Review feature importance quarterly',
            'Validate data quality upstream'
        ]
        
        return report

# ==================== DEMO ====================

def create_demo_data():
    '''Create demo data for monitoring'''
    np.random.seed(42)
    
    # Baseline data
    n_baseline = 1000
    baseline_features = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_baseline),
        'feature_2': np.random.normal(0, 1, n_baseline),
        'feature_3': np.random.uniform(0, 1, n_baseline)
    })
    
    # Current data with drift
    n_current = 500
    current_features = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, n_current),  # Mean shift
        'feature_2': np.random.normal(0, 1, n_current),       # No drift
        'feature_3': np.random.uniform(0.2, 1.2, n_current)   # Distribution shift
    })
    
    # Predictions and actuals
    predictions = np.random.randint(0, 2, n_current)
    actuals = np.random.randint(0, 2, n_current)
    
    return baseline_features, current_features, predictions, actuals

if __name__ == '__main__':
    print('🔍 MODEL MONITORING & DRIFT DETECTION - ULTRAPLATFORM')
    print('='*60)
    
    # Initialize monitoring system
    monitor = ModelMonitoringSystem()
    
    # Register model
    config = ModelMonitoringConfig(
        model_id='model_001',
        model_name='risk_classifier_v1',
        monitoring_frequency='hourly',
        drift_threshold=0.05,
        performance_threshold=0.05,
        alert_enabled=True,
        auto_retrain=False
    )
    
    # Get demo data
    baseline_data, current_data, predictions, actuals = create_demo_data()
    
    print('\n📝 Registering model for monitoring...')
    monitor.register_model(config, baseline_data)
    
    # Simulate predictions
    print('\n📊 Simulating predictions...')
    for i in range(len(predictions)):
        monitor.monitor_prediction(
            model_id='model_001',
            features=current_data.iloc[i].to_dict(),
            prediction=predictions[i],
            actual=actuals[i] if i % 2 == 0 else None  # Simulate delayed actuals
        )
    
    # Run monitoring check
    print('\n🔍 Running comprehensive monitoring check...')
    results = monitor.run_monitoring_check('model_001')
    
    # Generate report
    print('\n📄 Generating monitoring report...')
    report = monitor.generate_monitoring_report('model_001')
    
    print('\n✅ Monitoring system demonstration complete!')

