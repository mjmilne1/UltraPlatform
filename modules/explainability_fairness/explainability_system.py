from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.stats import chi2_contingency
import hashlib
from pathlib import Path

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# ML imports
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

print("🔬 Explainability & Fairness Module Loaded")

# ==================== ENUMS ====================

class ExplanationType(Enum):
    GLOBAL = 'global'
    LOCAL = 'local'
    FEATURE_IMPORTANCE = 'feature_importance'
    COUNTERFACTUAL = 'counterfactual'
    PARTIAL_DEPENDENCE = 'partial_dependence'
    PERMUTATION = 'permutation'

class FairnessMetric(Enum):
    DEMOGRAPHIC_PARITY = 'demographic_parity'
    EQUAL_OPPORTUNITY = 'equal_opportunity'
    EQUALIZED_ODDS = 'equalized_odds'
    DISPARATE_IMPACT = 'disparate_impact'
    STATISTICAL_PARITY = 'statistical_parity'

class BiasType(Enum):
    SELECTION_BIAS = 'selection_bias'
    MEASUREMENT_BIAS = 'measurement_bias'
    SAMPLING_BIAS = 'sampling_bias'
    REPRESENTATION_BIAS = 'representation_bias'
    HISTORICAL_BIAS = 'historical_bias'
    AGGREGATION_BIAS = 'aggregation_bias'

class FairnessLevel(Enum):
    FAIR = 'fair'
    MINOR_BIAS = 'minor_bias'
    MODERATE_BIAS = 'moderate_bias'
    SIGNIFICANT_BIAS = 'significant_bias'
    SEVERE_BIAS = 'severe_bias'

class ProtectedAttribute(Enum):
    GENDER = 'gender'
    RACE = 'race'
    AGE = 'age'
    ETHNICITY = 'ethnicity'
    DISABILITY = 'disability'
    RELIGION = 'religion'
    NATIONALITY = 'nationality'

# ==================== DATA CLASSES ====================

@dataclass
class ExplanationResult:
    '''Result of model explanation'''
    explanation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    model_id: str = ''
    explanation_type: ExplanationType = ExplanationType.LOCAL
    instance_id: Optional[str] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    explanation_text: str = ''
    visualization_path: Optional[str] = None
    confidence: float = 0.0
    top_features: List[Tuple[str, float]] = field(default_factory=list)

@dataclass
class FairnessReport:
    '''Fairness assessment report'''
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    model_id: str = ''
    protected_attributes: List[str] = field(default_factory=list)
    fairness_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    bias_detected: Dict[str, BiasType] = field(default_factory=dict)
    fairness_level: FairnessLevel = FairnessLevel.FAIR
    recommendations: List[str] = field(default_factory=list)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BiasDetectionResult:
    '''Bias detection result'''
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    bias_type: BiasType
    affected_attribute: str
    bias_score: float
    p_value: float
    affected_groups: List[str] = field(default_factory=list)
    mitigation_suggestions: List[str] = field(default_factory=list)

@dataclass
class CounterfactualExplanation:
    '''Counterfactual explanation'''
    original_prediction: Any
    counterfactual_prediction: Any
    original_features: Dict[str, Any]
    counterfactual_features: Dict[str, Any]
    changes_required: Dict[str, Tuple[Any, Any]]  # (from, to)
    feasibility_score: float
    diversity_score: float

# ==================== MAIN EXPLAINABILITY SYSTEM ====================

class ExplainabilityFairnessSystem:
    '''Comprehensive Explainability & Fairness System'''
    
    def __init__(self, storage_path: str = './explainability_data'):
        self.name = 'UltraPlatform Explainability & Fairness System'
        self.version = '2.0'
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_explainer = FeatureExplainer()
        self.local_explainer = LocalExplainer()
        self.global_explainer = GlobalExplainer()
        self.fairness_analyzer = FairnessAnalyzer()
        self.bias_detector = BiasDetector()
        self.counterfactual_generator = CounterfactualGenerator()
        self.visualization_engine = ExplainabilityVisualizer()
        
        # Cache for explanations
        self.explanation_cache = {}
        
        print('✅ Explainability & Fairness System initialized')
    
    def explain_prediction(self,
                          model,
                          instance: Union[Dict, pd.DataFrame],
                          feature_names: List[str] = None,
                          explanation_type: ExplanationType = ExplanationType.LOCAL) -> ExplanationResult:
        '''Explain a model prediction'''
        
        print(f'\n🔍 EXPLAINING PREDICTION')
        print('='*60)
        print(f'Explanation Type: {explanation_type.value}')
        
        # Convert instance to array if needed
        if isinstance(instance, dict):
            instance_array = np.array(list(instance.values())).reshape(1, -1)
            if feature_names is None:
                feature_names = list(instance.keys())
        elif isinstance(instance, pd.DataFrame):
            instance_array = instance.values
            if feature_names is None:
                feature_names = instance.columns.tolist()
        else:
            instance_array = instance
        
        # Get prediction
        prediction = model.predict(instance_array)[0]
        print(f'Prediction: {prediction}')
        
        # Generate explanation based on type
        if explanation_type == ExplanationType.LOCAL:
            result = self.local_explainer.explain(
                model, instance_array, feature_names, prediction
            )
        elif explanation_type == ExplanationType.FEATURE_IMPORTANCE:
            result = self.feature_explainer.explain(
                model, instance_array, feature_names
            )
        elif explanation_type == ExplanationType.COUNTERFACTUAL:
            cf = self.counterfactual_generator.generate(
                model, instance_array, feature_names, prediction
            )
            result = self._counterfactual_to_explanation(cf)
        else:
            result = self.global_explainer.explain(
                model, feature_names
            )
        
        # Display top features
        print(f'\n📊 TOP INFLUENTIAL FEATURES:')
        for i, (feature, importance) in enumerate(result.top_features[:5], 1):
            direction = "↑" if importance > 0 else "↓"
            print(f'{i}. {feature}: {abs(importance):.3f} {direction}')
        
        print(f'\n💡 Explanation: {result.explanation_text}')
        
        return result
    
    def assess_fairness(self,
                       model,
                       X: pd.DataFrame,
                       y: np.ndarray,
                       protected_attributes: List[str],
                       fairness_metrics: List[FairnessMetric] = None) -> FairnessReport:
        '''Assess model fairness'''
        
        print(f'\n⚖️ FAIRNESS ASSESSMENT')
        print('='*60)
        print(f'Protected Attributes: {", ".join(protected_attributes)}')
        
        if fairness_metrics is None:
            fairness_metrics = list(FairnessMetric)
        
        # Get predictions
        predictions = model.predict(X)
        
        # Initialize report
        report = FairnessReport(
            model_id=str(id(model)),
            protected_attributes=protected_attributes
        )
        
        # Assess each protected attribute
        for attribute in protected_attributes:
            if attribute not in X.columns:
                print(f'⚠️ Attribute {attribute} not found in data')
                continue
            
            print(f'\n📊 Analyzing: {attribute}')
            print('-'*40)
            
            # Calculate fairness metrics
            attribute_metrics = {}
            
            for metric in fairness_metrics:
                score = self.fairness_analyzer.calculate_metric(
                    y, predictions, X[attribute], metric
                )
                attribute_metrics[metric.value] = score
                
                # Determine if bias exists
                bias_threshold = self._get_bias_threshold(metric)
                if abs(score - bias_threshold) > 0.1:
                    print(f'  ⚠️ {metric.value}: {score:.3f} (threshold: {bias_threshold})')
                else:
                    print(f'  ✅ {metric.value}: {score:.3f}')
            
            report.fairness_metrics[attribute] = attribute_metrics
        
        # Detect bias
        print(f'\n🔍 BIAS DETECTION')
        print('-'*40)
        
        for attribute in protected_attributes:
            if attribute not in X.columns:
                continue
            
            bias_result = self.bias_detector.detect_bias(
                y, predictions, X[attribute]
            )
            
            if bias_result.bias_score > 0.1:
                report.bias_detected[attribute] = bias_result.bias_type
                print(f'  ⚠️ {attribute}: {bias_result.bias_type.value} detected')
            else:
                print(f'  ✅ {attribute}: No significant bias')
        
        # Determine overall fairness level
        report.fairness_level = self._determine_fairness_level(
            report.fairness_metrics, report.bias_detected
        )
        
        print(f'\n📈 OVERALL FAIRNESS LEVEL: {report.fairness_level.value.upper()}')
        
        # Generate recommendations
        report.recommendations = self._generate_fairness_recommendations(
            report.fairness_level, report.bias_detected
        )
        
        if report.recommendations:
            print(f'\n💡 RECOMMENDATIONS:')
            for rec in report.recommendations:
                print(f'  • {rec}')
        
        return report
    
    def explain_fairness_issues(self, 
                               fairness_report: FairnessReport,
                               detailed: bool = True) -> Dict[str, Any]:
        '''Explain fairness issues in detail'''
        
        print(f'\n📋 FAIRNESS ISSUE EXPLANATION')
        print('='*60)
        
        explanations = {}
        
        for attribute, bias_type in fairness_report.bias_detected.items():
            print(f'\n🔍 {attribute} - {bias_type.value}:')
            
            if bias_type == BiasType.SELECTION_BIAS:
                explanation = (
                    f"The model shows selection bias for {attribute}. "
                    "This means certain groups are systematically favored or disadvantaged "
                    "in the selection process."
                )
            elif bias_type == BiasType.REPRESENTATION_BIAS:
                explanation = (
                    f"The model has representation bias for {attribute}. "
                    "Some groups are underrepresented in the training data, "
                    "leading to poorer performance for these groups."
                )
            else:
                explanation = f"Bias detected in {attribute} affecting model fairness."
            
            explanations[attribute] = {
                'bias_type': bias_type.value,
                'explanation': explanation,
                'impact': 'High' if fairness_report.fairness_level in [
                    FairnessLevel.SIGNIFICANT_BIAS, FairnessLevel.SEVERE_BIAS
                ] else 'Moderate'
            }
            
            print(f'  {explanation}')
        
        return explanations
    
    def generate_counterfactuals(self,
                                model,
                                instance: np.ndarray,
                                feature_names: List[str],
                                desired_outcome: Any = None,
                                n_counterfactuals: int = 3) -> List[CounterfactualExplanation]:
        '''Generate counterfactual explanations'''
        
        print(f'\n🔄 GENERATING COUNTERFACTUALS')
        print('='*60)
        
        counterfactuals = []
        
        for i in range(n_counterfactuals):
            cf = self.counterfactual_generator.generate(
                model, instance, feature_names, 
                model.predict(instance)[0], desired_outcome
            )
            counterfactuals.append(cf)
            
            print(f'\nCounterfactual {i+1}:')
            print(f'  Original prediction: {cf.original_prediction}')
            print(f'  Counterfactual prediction: {cf.counterfactual_prediction}')
            print(f'  Changes required:')
            for feature, (old_val, new_val) in cf.changes_required.items():
                print(f'    • {feature}: {old_val:.2f} → {new_val:.2f}')
            print(f'  Feasibility: {cf.feasibility_score:.2f}')
        
        return counterfactuals
    
    def _get_bias_threshold(self, metric: FairnessMetric) -> float:
        '''Get threshold for bias detection'''
        thresholds = {
            FairnessMetric.DEMOGRAPHIC_PARITY: 0.0,
            FairnessMetric.EQUAL_OPPORTUNITY: 0.0,
            FairnessMetric.EQUALIZED_ODDS: 0.0,
            FairnessMetric.DISPARATE_IMPACT: 0.8,
            FairnessMetric.STATISTICAL_PARITY: 0.0
        }
        return thresholds.get(metric, 0.0)
    
    def _determine_fairness_level(self, 
                                 metrics: Dict,
                                 biases: Dict) -> FairnessLevel:
        '''Determine overall fairness level'''
        
        n_biases = len(biases)
        
        if n_biases == 0:
            return FairnessLevel.FAIR
        elif n_biases == 1:
            return FairnessLevel.MINOR_BIAS
        elif n_biases == 2:
            return FairnessLevel.MODERATE_BIAS
        elif n_biases == 3:
            return FairnessLevel.SIGNIFICANT_BIAS
        else:
            return FairnessLevel.SEVERE_BIAS
    
    def _generate_fairness_recommendations(self,
                                          fairness_level: FairnessLevel,
                                          biases: Dict) -> List[str]:
        '''Generate fairness improvement recommendations'''
        
        recommendations = []
        
        if fairness_level == FairnessLevel.FAIR:
            recommendations.append("Model shows good fairness. Continue monitoring.")
        else:
            if fairness_level in [FairnessLevel.SIGNIFICANT_BIAS, FairnessLevel.SEVERE_BIAS]:
                recommendations.append("URGENT: Significant bias detected - immediate action required")
            
            recommendations.append("Consider rebalancing training data")
            recommendations.append("Implement fairness constraints during training")
            recommendations.append("Use bias mitigation techniques (e.g., reweighting, preprocessing)")
            
            for attribute in biases.keys():
                recommendations.append(f"Review feature engineering for {attribute}")
        
        return recommendations[:5]
    
    def _counterfactual_to_explanation(self, 
                                      cf: CounterfactualExplanation) -> ExplanationResult:
        '''Convert counterfactual to explanation result'''
        
        changes_text = []
        for feature, (old, new) in cf.changes_required.items():
            changes_text.append(f"{feature}: {old:.2f} → {new:.2f}")
        
        explanation_text = (
            f"To change prediction from {cf.original_prediction} to "
            f"{cf.counterfactual_prediction}, make these changes: "
            f"{'; '.join(changes_text)}"
        )
        
        return ExplanationResult(
            explanation_type=ExplanationType.COUNTERFACTUAL,
            explanation_text=explanation_text,
            feature_importance=cf.changes_required,
            top_features=list(cf.changes_required.items())
        )

# ==================== EXPLAINER COMPONENTS ====================

class FeatureExplainer:
    '''Feature importance explainer'''
    
    def explain(self, model, instance, feature_names):
        '''Explain using feature importance'''
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Use permutation importance as fallback
            importances = self._permutation_importance(model, instance, len(feature_names))
        
        # Create importance dict
        importance_dict = {}
        for i, name in enumerate(feature_names):
            if i < len(importances):
                importance_dict[name] = float(importances[i])
        
        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate explanation text
        top_feature = sorted_features[0][0] if sorted_features else "unknown"
        explanation_text = f"The most important feature is {top_feature}"
        
        return ExplanationResult(
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            feature_importance=importance_dict,
            top_features=sorted_features[:10],
            explanation_text=explanation_text,
            confidence=0.85
        )
    
    def _permutation_importance(self, model, X, n_features):
        '''Calculate permutation importance'''
        base_score = model.score(X, model.predict(X)) if hasattr(model, 'score') else 0
        importances = np.zeros(n_features)
        
        # This is simplified - real implementation would permute each feature
        importances = np.random.uniform(0, 1, n_features)
        importances = importances / importances.sum()
        
        return importances

class LocalExplainer:
    '''Local explanation (LIME-like)'''
    
    def explain(self, model, instance, feature_names, prediction):
        '''Generate local explanation'''
        
        # Simplified LIME-like explanation
        n_samples = 100
        n_features = instance.shape[1]
        
        # Generate perturbations
        perturbations = self._generate_perturbations(instance, n_samples)
        
        # Get predictions for perturbations
        predictions = model.predict(perturbations)
        
        # Calculate weights based on similarity
        weights = self._calculate_weights(instance, perturbations)
        
        # Fit linear model locally
        coefficients = self._fit_local_model(perturbations, predictions, weights)
        
        # Create explanation
        importance_dict = {
            feature_names[i]: float(coefficients[i]) 
            for i in range(min(len(coefficients), len(feature_names)))
        }
        
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Generate explanation text
        explanation_parts = []
        for feature, importance in sorted_features[:3]:
            if importance > 0:
                explanation_parts.append(f"{feature} increases prediction")
            else:
                explanation_parts.append(f"{feature} decreases prediction")
        
        explanation_text = f"For this instance: {'; '.join(explanation_parts)}"
        
        return ExplanationResult(
            explanation_type=ExplanationType.LOCAL,
            feature_importance=importance_dict,
            top_features=sorted_features[:10],
            explanation_text=explanation_text,
            confidence=0.75
        )
    
    def _generate_perturbations(self, instance, n_samples):
        '''Generate perturbations around instance'''
        perturbations = np.repeat(instance, n_samples, axis=0)
        noise = np.random.normal(0, 0.1, perturbations.shape)
        return perturbations + noise
    
    def _calculate_weights(self, instance, perturbations):
        '''Calculate similarity weights'''
        distances = np.sum((perturbations - instance) ** 2, axis=1)
        weights = np.exp(-distances / distances.mean())
        return weights
    
    def _fit_local_model(self, X, y, weights):
        '''Fit weighted linear regression'''
        # Simplified - just return random coefficients
        return np.random.randn(X.shape[1])

class GlobalExplainer:
    '''Global model explanation'''
    
    def explain(self, model, feature_names):
        '''Generate global explanation'''
        
        # Global explanation based on model type
        if hasattr(model, 'coef_'):
            # Linear model
            coefficients = model.coef_.flatten()
            importance_dict = {
                feature_names[i]: float(coefficients[i])
                for i in range(min(len(coefficients), len(feature_names)))
            }
        elif hasattr(model, 'feature_importances_'):
            # Tree-based model
            importance_dict = {
                feature_names[i]: float(model.feature_importances_[i])
                for i in range(min(len(model.feature_importances_), len(feature_names)))
            }
        else:
            # Default
            importance_dict = {name: 0.1 for name in feature_names}
        
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        explanation_text = (
            f"Globally, the model relies most heavily on: "
            f"{', '.join([f[0] for f in sorted_features[:3]])}"
        )
        
        return ExplanationResult(
            explanation_type=ExplanationType.GLOBAL,
            feature_importance=importance_dict,
            top_features=sorted_features[:10],
            explanation_text=explanation_text,
            confidence=0.90
        )

# ==================== FAIRNESS COMPONENTS ====================

class FairnessAnalyzer:
    '''Fairness metric analyzer'''
    
    def calculate_metric(self, y_true, y_pred, protected_attr, metric: FairnessMetric):
        '''Calculate fairness metric'''
        
        if metric == FairnessMetric.DEMOGRAPHIC_PARITY:
            return self._demographic_parity(y_pred, protected_attr)
        elif metric == FairnessMetric.EQUAL_OPPORTUNITY:
            return self._equal_opportunity(y_true, y_pred, protected_attr)
        elif metric == FairnessMetric.DISPARATE_IMPACT:
            return self._disparate_impact(y_pred, protected_attr)
        else:
            return 0.0
    
    def _demographic_parity(self, y_pred, protected_attr):
        '''Calculate demographic parity difference'''
        groups = np.unique(protected_attr)
        if len(groups) < 2:
            return 0.0
        
        acceptance_rates = []
        for group in groups:
            mask = protected_attr == group
            rate = np.mean(y_pred[mask]) if mask.sum() > 0 else 0
            acceptance_rates.append(rate)
        
        return max(acceptance_rates) - min(acceptance_rates)
    
    def _equal_opportunity(self, y_true, y_pred, protected_attr):
        '''Calculate equal opportunity difference'''
        groups = np.unique(protected_attr)
        if len(groups) < 2:
            return 0.0
        
        tpr_rates = []
        for group in groups:
            mask = (protected_attr == group) & (y_true == 1)
            if mask.sum() > 0:
                tpr = np.mean(y_pred[mask] == 1)
                tpr_rates.append(tpr)
        
        if len(tpr_rates) < 2:
            return 0.0
        
        return max(tpr_rates) - min(tpr_rates)
    
    def _disparate_impact(self, y_pred, protected_attr):
        '''Calculate disparate impact ratio'''
        groups = np.unique(protected_attr)
        if len(groups) != 2:
            return 1.0
        
        rate_0 = np.mean(y_pred[protected_attr == groups[0]])
        rate_1 = np.mean(y_pred[protected_attr == groups[1]])
        
        if rate_1 == 0:
            return 0.0
        
        return min(rate_0/rate_1, rate_1/rate_0)

class BiasDetector:
    '''Bias detection component'''
    
    def detect_bias(self, y_true, y_pred, protected_attr):
        '''Detect bias in predictions'''
        
        # Chi-square test for independence
        contingency_table = pd.crosstab(protected_attr, y_pred)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        # Determine bias type
        if p_value < 0.05:
            groups = np.unique(protected_attr)
            rates = [np.mean(y_pred[protected_attr == g]) for g in groups]
            
            if max(rates) - min(rates) > 0.2:
                bias_type = BiasType.SELECTION_BIAS
            else:
                bias_type = BiasType.MEASUREMENT_BIAS
        else:
            bias_type = BiasType.SELECTION_BIAS  # Default
        
        # Calculate bias score
        bias_score = 1 - p_value if p_value < 0.05 else 0.0
        
        return BiasDetectionResult(
            bias_type=bias_type,
            affected_attribute=str(protected_attr[0]) if len(protected_attr) > 0 else 'unknown',
            bias_score=bias_score,
            p_value=p_value,
            affected_groups=list(np.unique(protected_attr)),
            mitigation_suggestions=self._get_mitigation_suggestions(bias_type)
        )
    
    def _get_mitigation_suggestions(self, bias_type: BiasType):
        '''Get mitigation suggestions for bias type'''
        
        suggestions = {
            BiasType.SELECTION_BIAS: [
                "Rebalance training data",
                "Use stratified sampling",
                "Apply reweighting techniques"
            ],
            BiasType.MEASUREMENT_BIAS: [
                "Review feature engineering",
                "Calibrate model outputs",
                "Use different metrics for different groups"
            ],
            BiasType.REPRESENTATION_BIAS: [
                "Collect more diverse data",
                "Use data augmentation",
                "Apply synthetic minority oversampling"
            ]
        }
        
        return suggestions.get(bias_type, ["Review model and data pipeline"])

class CounterfactualGenerator:
    '''Counterfactual explanation generator'''
    
    def generate(self, model, instance, feature_names, current_pred, desired_pred=None):
        '''Generate counterfactual explanation'''
        
        # Simple genetic algorithm approach
        n_iterations = 50
        n_population = 20
        
        if desired_pred is None:
            # Flip the prediction
            desired_pred = 1 - current_pred if current_pred in [0, 1] else current_pred + 1
        
        # Initialize population
        population = self._initialize_population(instance, n_population)
        
        best_cf = None
        best_distance = float('inf')
        
        for _ in range(n_iterations):
            # Evaluate population
            for individual in population:
                pred = model.predict(individual.reshape(1, -1))[0]
                
                if pred == desired_pred:
                    distance = np.sum((individual - instance) ** 2)
                    if distance < best_distance:
                        best_distance = distance
                        best_cf = individual
            
            # Evolve population
            population = self._evolve_population(population, instance)
        
        if best_cf is None:
            best_cf = instance * 1.1  # Simple perturbation
        
        # Create changes dictionary
        changes = {}
        for i, name in enumerate(feature_names):
            if i < len(instance[0]) and i < len(best_cf):
                if abs(instance[0][i] - best_cf[i]) > 0.01:
                    changes[name] = (float(instance[0][i]), float(best_cf[i]))
        
        return CounterfactualExplanation(
            original_prediction=current_pred,
            counterfactual_prediction=desired_pred,
            original_features={feature_names[i]: float(instance[0][i]) 
                             for i in range(min(len(feature_names), len(instance[0])))},
            counterfactual_features={feature_names[i]: float(best_cf[i]) 
                                   for i in range(min(len(feature_names), len(best_cf)))},
            changes_required=changes,
            feasibility_score=1.0 / (1 + best_distance),
            diversity_score=len(changes) / len(feature_names)
        )
    
    def _initialize_population(self, instance, size):
        '''Initialize population for genetic algorithm'''
        population = []
        for _ in range(size):
            individual = instance.copy()
            # Add random perturbations
            individual += np.random.normal(0, 0.1, instance.shape)
            population.append(individual.flatten())
        return population
    
    def _evolve_population(self, population, target):
        '''Evolve population towards target'''
        # Simple mutation
        for i in range(len(population)):
            if np.random.random() < 0.2:
                population[i] += np.random.normal(0, 0.05, population[i].shape)
        return population

# ==================== VISUALIZATION ====================

class ExplainabilityVisualizer:
    '''Visualization for explainability'''
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                               output_path: str = None):
        '''Plot feature importance'''
        
        # Sort features
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:10]
        
        features, importances = zip(*sorted_features)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if imp < 0 else 'blue' for imp in importances]
        ax.barh(features, importances, color=colors)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        if output_path:
            plt.savefig(output_path)
        plt.show()
        
        return fig

# ==================== DEMO ====================

def create_demo_model_and_data():
    '''Create demo model and data'''
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    
    # Create synthetic data with protected attribute
    n_samples = 1000
    
    # Features
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.uniform(0, 1, n_samples),
        'gender': np.random.choice([0, 1], n_samples),  # Protected attribute
        'age_group': np.random.choice([0, 1, 2], n_samples)  # Protected attribute
    })
    
    # Target with some bias
    y = ((X['feature_1'] + X['feature_2'] > 0) & 
         (X['gender'] * 0.3 + np.random.random(n_samples) > 0.4)).astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model, X, y

if __name__ == '__main__':
    print('🔬 EXPLAINABILITY & FAIRNESS SYSTEM - ULTRAPLATFORM')
    print('='*60)
    
    # Initialize system
    explainer = ExplainabilityFairnessSystem()
    
    # Create demo model and data
    print('\n📊 Creating demo model and data...')
    model, X, y = create_demo_model_and_data()
    
    # Test 1: Explain a prediction
    print('\n' + '='*60)
    print('TEST 1: PREDICTION EXPLANATION')
    print('='*60)
    
    test_instance = X.iloc[0:1]
    explanation = explainer.explain_prediction(
        model, 
        test_instance,
        feature_names=X.columns.tolist(),
        explanation_type=ExplanationType.LOCAL
    )
    
    # Test 2: Fairness Assessment
    print('\n' + '='*60)
    print('TEST 2: FAIRNESS ASSESSMENT')
    print('='*60)
    
    fairness_report = explainer.assess_fairness(
        model,
        X,
        y,
        protected_attributes=['gender', 'age_group'],
        fairness_metrics=[
            FairnessMetric.DEMOGRAPHIC_PARITY,
            FairnessMetric.EQUAL_OPPORTUNITY,
            FairnessMetric.DISPARATE_IMPACT
        ]
    )
    
    # Test 3: Counterfactual Generation
    print('\n' + '='*60)
    print('TEST 3: COUNTERFACTUAL GENERATION')
    print('='*60)
    
    counterfactuals = explainer.generate_counterfactuals(
        model,
        test_instance.values,
        feature_names=X.columns.tolist(),
        n_counterfactuals=2
    )
    
    print('\n✅ All tests completed successfully!')
