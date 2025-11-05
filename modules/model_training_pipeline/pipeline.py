from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import json
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

print("🤖 Model Training Pipeline Module Loaded")

# ==================== ENUMS ====================

class ModelType(Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'

class ModelStatus(Enum):
    TRAINING = 'training'
    VALIDATING = 'validating'
    STAGED = 'staged'
    PRODUCTION = 'production'
    ARCHIVED = 'archived'

# ==================== DATA CLASSES ====================

@dataclass
class ModelConfig:
    model_name: str
    model_type: ModelType
    algorithm: str
    features: List[str] = field(default_factory=list)
    target: str = 'target'
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    version: str = '1.0.0'

@dataclass
class TrainingConfig:
    train_size: float = 0.7
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    cv_folds: int = 5
    hyperparameter_tuning: bool = True

@dataclass
class ModelMetrics:
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0

# ==================== MAIN PIPELINE ====================

class ModelTrainingPipeline:
    def __init__(self, base_path: str = './models'):
        self.name = 'UltraPlatform Model Training Pipeline'
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        print('✅ Model Training Pipeline initialized')
    
    def train_model(self, data: pd.DataFrame, model_config: ModelConfig, training_config: TrainingConfig = None):
        print('\n' + '='*60)
        print('MODEL TRAINING PIPELINE')
        print('='*60)
        print(f'Model: {model_config.model_name}')
        print(f'Type: {model_config.model_type.value}')
        print(f'Algorithm: {model_config.algorithm}')
        
        if training_config is None:
            training_config = TrainingConfig()
        
        # Step 1: Prepare data
        print('\n1️⃣ DATA PREPARATION')
        X = data[model_config.features] if model_config.features else data.drop(columns=[model_config.target])
        y = data[model_config.target]
        print(f'   Samples: {len(X)}')
        print(f'   Features: {X.shape[1]}')
        
        # Step 2: Split data
        print('\n2️⃣ DATA SPLITTING')
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - training_config.train_size), 
            random_state=training_config.random_state
        )
        
        val_size = training_config.validation_size / (training_config.validation_size + training_config.test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size),
            random_state=training_config.random_state
        )
        
        print(f'   Train: {len(X_train)} samples')
        print(f'   Val: {len(X_val)} samples')
        print(f'   Test: {len(X_test)} samples')
        
        # Step 3: Scale features
        print('\n3️⃣ FEATURE SCALING')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        print('   ✅ Features scaled')
        
        # Step 4: Train model
        print('\n4️⃣ MODEL TRAINING')
        model = self._get_model(model_config)
        
        if training_config.hyperparameter_tuning and model_config.model_type == ModelType.CLASSIFICATION:
            print('   Running hyperparameter tuning...')
            param_grid = self._get_param_grid(model_config.algorithm)
            
            grid_search = GridSearchCV(
                model, param_grid, 
                cv=training_config.cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f'   Best params: {grid_search.best_params_}')
        else:
            model.fit(X_train, y_train)
        
        print('   ✅ Model trained')
        
        # Step 5: Evaluate
        print('\n5️⃣ MODEL EVALUATION')
        metrics = self._evaluate_model(model, X_test, y_test, model_config.model_type)
        print(f'   Accuracy: {metrics.accuracy:.3f}')
        print(f'   Precision: {metrics.precision:.3f}')
        print(f'   Recall: {metrics.recall:.3f}')
        print(f'   F1 Score: {metrics.f1_score:.3f}')
        if hasattr(metrics, 'auc_roc'):
            print(f'   AUC-ROC: {metrics.auc_roc:.3f}')
        
        # Step 6: Save model
        print('\n6️⃣ SAVING MODEL')
        model_path = self.base_path / f'{model_config.model_name}_{metrics.model_id}.pkl'
        joblib.dump(model, model_path)
        print(f'   ✅ Model saved to: {model_path}')
        
        print('\n' + '='*60)
        print('✅ TRAINING COMPLETE!')
        print('='*60)
        
        return model, metrics, str(model_path)
    
    def _get_model(self, config: ModelConfig):
        algorithm = config.algorithm.lower()
        
        if 'logistic' in algorithm:
            return LogisticRegression(max_iter=1000, random_state=42)
        elif 'forest' in algorithm:
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif 'gradient' in algorithm:
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _get_param_grid(self, algorithm: str):
        algorithm = algorithm.lower()
        
        if 'logistic' in algorithm:
            return {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }
        elif 'forest' in algorithm:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        else:
            return {}
    
    def _evaluate_model(self, model, X_test, y_test, model_type: ModelType):
        y_pred = model.predict(X_test)
        
        metrics = ModelMetrics()
        
        if model_type == ModelType.CLASSIFICATION:
            metrics.accuracy = accuracy_score(y_test, y_pred)
            metrics.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics.f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:  # Binary classification
                    metrics.auc_roc = roc_auc_score(y_test, y_proba[:, 1])
        
        return metrics

# ==================== DEMO ====================

def create_sample_data(n_samples=1000):
    '''Create sample data for demonstration'''
    np.random.seed(42)
    
    # Create features
    X = np.random.randn(n_samples, 5)
    
    # Create target
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    return df

if __name__ == '__main__':
    print('🤖 MODEL TRAINING PIPELINE - ULTRAPLATFORM')
    print('='*60)
    
    # Create sample data
    print('\n📊 Creating sample dataset...')
    data = create_sample_data(1000)
    print(f'Dataset shape: {data.shape}')
    
    # Configure model
    model_config = ModelConfig(
        model_name='risk_classifier',
        model_type=ModelType.CLASSIFICATION,
        algorithm='random_forest',
        target='target'
    )
    
    # Train model
    print('\n🚀 Starting model training...')
    pipeline = ModelTrainingPipeline()
    model, metrics, path = pipeline.train_model(data, model_config)
    
    print('\n✅ Pipeline execution complete!')
