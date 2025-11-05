from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import math
import hashlib
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif

class FeatureType(Enum):
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'
    TEMPORAL = 'temporal'
    TEXT = 'text'
    TECHNICAL = 'technical'
    STATISTICAL = 'statistical'
    ENGINEERED = 'engineered'
    INTERACTION = 'interaction'
    EMBEDDING = 'embedding'

class TransformationType(Enum):
    SCALING = 'scaling'
    NORMALIZATION = 'normalization'
    ENCODING = 'encoding'
    BINNING = 'binning'
    POLYNOMIAL = 'polynomial'
    LOG_TRANSFORM = 'log_transform'
    DIFFERENCING = 'differencing'
    AGGREGATION = 'aggregation'
    DECOMPOSITION = 'decomposition'

class DataQuality(Enum):
    EXCELLENT = 'excellent'
    GOOD = 'good'
    FAIR = 'fair'
    POOR = 'poor'
    UNUSABLE = 'unusable'

@dataclass
class Feature:
    '''Individual feature definition'''
    name: str
    feature_type: FeatureType
    importance: float = 0.0
    missing_ratio: float = 0.0
    cardinality: int = 0
    statistics: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    quality: DataQuality = DataQuality.GOOD
    
    def to_dict(self):
        return {
            'name': self.name,
            'type': self.feature_type.value,
            'importance': self.importance,
            'missing_ratio': self.missing_ratio,
            'quality': self.quality.value
        }

@dataclass
class FeatureSet:
    '''Collection of features'''
    feature_set_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ''
    features: List[Feature] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    version: str = '1.0'
    
    def get_feature_names(self):
        return [f.name for f in self.features]

class FeatureEngineeringPipeline:
    '''Comprehensive Feature Engineering Pipeline for UltraPlatform'''
    
    def __init__(self):
        self.name = 'UltraPlatform Feature Engineering'
        self.version = '2.0'
        
        # Core components
        self.feature_extractor = FeatureExtractor()
        self.feature_transformer = FeatureTransformer()
        self.feature_generator = FeatureGenerator()
        self.feature_selector = FeatureSelector()
        self.feature_store = FeatureStore()
        self.feature_validator = FeatureValidator()
        
        # Specialized processors
        self.technical_features = TechnicalFeatureGenerator()
        self.statistical_features = StatisticalFeatureGenerator()
        self.temporal_features = TemporalFeatureGenerator()
        self.text_features = TextFeatureGenerator()
        self.interaction_features = InteractionFeatureGenerator()
        
        print('🔧 Feature Engineering Pipeline initialized')
    
    def process_data(self, data: pd.DataFrame, target: Optional[str] = None, 
                     feature_config: Optional[Dict] = None):
        '''Process data through complete feature engineering pipeline'''
        
        print('FEATURE ENGINEERING PIPELINE')
        print('='*80)
        print(f'Input Shape: {data.shape}')
        print(f'Target Variable: {target if target else "None"}')
        print()
        
        # Step 1: Data Quality Assessment
        print('1️⃣ DATA QUALITY ASSESSMENT')
        print('-'*40)
        quality_report = self.feature_validator.assess_quality(data)
        print(f'  Overall Quality: {quality_report["overall_quality"].value}')
        print(f'  Missing Data: {quality_report["missing_percentage"]:.1f}%')
        print(f'  Duplicates: {quality_report["duplicate_rows"]}')
        print(f'  Outliers: {quality_report["outlier_percentage"]:.1f}%')
        print(f'  Data Types: {quality_report["data_types"]}')
        
        # Step 2: Feature Extraction
        print('\n2️⃣ FEATURE EXTRACTION')
        print('-'*40)
        extracted = self.feature_extractor.extract_features(data)
        print(f'  Numerical Features: {len(extracted["numerical"])}')
        print(f'  Categorical Features: {len(extracted["categorical"])}')
        print(f'  Datetime Features: {len(extracted["datetime"])}')
        print(f'  Text Features: {len(extracted["text"])}')
        
        # Step 3: Feature Generation
        print('\n3️⃣ FEATURE GENERATION')
        print('-'*40)
        
        # Technical indicators
        if self._has_price_data(data):
            tech_features = self.technical_features.generate(data)
            print(f'  Technical Indicators: {len(tech_features)} generated')
        else:
            tech_features = pd.DataFrame()
            print(f'  Technical Indicators: Skipped (no price data)')
        
        # Statistical features
        stat_features = self.statistical_features.generate(data)
        print(f'  Statistical Features: {len(stat_features.columns)} generated')
        
        # Temporal features
        if extracted["datetime"]:
            temp_features = self.temporal_features.generate(data, extracted["datetime"])
            print(f'  Temporal Features: {len(temp_features.columns)} generated')
        else:
            temp_features = pd.DataFrame()
            print(f'  Temporal Features: Skipped (no datetime)')
        
        # Interaction features
        interact_features = self.interaction_features.generate(data, max_interactions=20)
        print(f'  Interaction Features: {len(interact_features.columns)} generated')
        
        # Step 4: Feature Transformation
        print('\n4️⃣ FEATURE TRANSFORMATION')
        print('-'*40)
        transformed = self.feature_transformer.transform_features(
            data, extracted["numerical"], extracted["categorical"]
        )
        print(f'  Scaled Features: {transformed["scaled_count"]}')
        print(f'  Encoded Features: {transformed["encoded_count"]}')
        print(f'  Binned Features: {transformed["binned_count"]}')
        print(f'  Log Transformed: {transformed["log_transformed_count"]}')
        
        # Step 5: Feature Selection
        print('\n5️⃣ FEATURE SELECTION')
        print('-'*40)
        
        # Combine all features
        all_features = self._combine_features([
            data, tech_features, stat_features, 
            temp_features, interact_features, transformed["data"]
        ])
        
        # Select best features
        if target and target in all_features.columns:
            selected = self.feature_selector.select_features(
                all_features.drop(columns=[target]), 
                all_features[target],
                k=50  # Top 50 features
            )
            print(f'  Initial Features: {len(all_features.columns)}')
            print(f'  Selected Features: {len(selected["selected_features"])}')
            print(f'  Selection Method: {selected["method"]}')
            print(f'\n  Top 10 Features:')
            for i, (feat, score) in enumerate(selected["feature_scores"][:10], 1):
                print(f'    {i}. {feat}: {score:.4f}')
        else:
            selected = {"selected_features": all_features.columns.tolist()}
            print(f'  Features Selected: {len(all_features.columns)} (no target for selection)')
        
        # Step 6: Feature Validation
        print('\n6️⃣ FEATURE VALIDATION')
        print('-'*40)
        validation = self.feature_validator.validate_features(
            all_features[selected["selected_features"]]
        )
        print(f'  Valid Features: {validation["valid_count"]}')
        print(f'  Invalid Features: {validation["invalid_count"]}')
        print(f'  Warnings: {len(validation["warnings"])}')
        if validation["warnings"]:
            for warning in validation["warnings"][:3]:
                print(f'    ⚠️ {warning}')
        
        # Step 7: Feature Storage
        print('\n7️⃣ FEATURE STORAGE')
        print('-'*40)
        feature_set = self._create_feature_set(
            selected["selected_features"],
            selected.get("feature_scores", [])
        )
        stored = self.feature_store.store_features(feature_set)
        print(f'  Feature Set ID: {stored["feature_set_id"]}')
        print(f'  Storage Location: {stored["location"]}')
        print(f'  Version: {stored["version"]}')
        
        # Return processed features
        return {
            'data': all_features[selected["selected_features"]],
            'feature_set': feature_set,
            'quality_report': quality_report,
            'statistics': self._calculate_statistics(all_features)
        }
    
    def _has_price_data(self, data):
        '''Check if data has price columns'''
        price_columns = ['close', 'open', 'high', 'low', 'price']
        return any(col in data.columns.str.lower() for col in price_columns)
    
    def _combine_features(self, dataframes):
        '''Combine multiple feature dataframes'''
        combined = pd.DataFrame()
        
        for df in dataframes:
            if isinstance(df, pd.DataFrame) and not df.empty:
                if combined.empty:
                    combined = df
                else:
                    # Avoid duplicate columns
                    new_cols = [col for col in df.columns if col not in combined.columns]
                    if new_cols:
                        combined = pd.concat([combined, df[new_cols]], axis=1)
        
        return combined
    
    def _create_feature_set(self, feature_names, feature_scores):
        '''Create feature set object'''
        features = []
        
        score_dict = dict(feature_scores) if feature_scores else {}
        
        for name in feature_names:
            feature = Feature(
                name=name,
                feature_type=self._infer_feature_type(name),
                importance=score_dict.get(name, 0.0)
            )
            features.append(feature)
        
        return FeatureSet(
            name=f'feature_set_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            features=features
        )
    
    def _infer_feature_type(self, feature_name):
        '''Infer feature type from name'''
        name_lower = feature_name.lower()
        
        if any(ind in name_lower for ind in ['sma', 'ema', 'rsi', 'macd', 'bb']):
            return FeatureType.TECHNICAL
        elif any(stat in name_lower for stat in ['mean', 'std', 'var', 'skew', 'kurt']):
            return FeatureType.STATISTICAL
        elif any(temp in name_lower for temp in ['hour', 'day', 'week', 'month', 'year']):
            return FeatureType.TEMPORAL
        elif '_x_' in name_lower or '_interaction' in name_lower:
            return FeatureType.INTERACTION
        else:
            return FeatureType.ENGINEERED
    
    def _calculate_statistics(self, data):
        '''Calculate feature statistics'''
        stats = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median(),
                'skewness': data[col].skew(),
                'kurtosis': data[col].kurtosis()
            }
        
        return stats

class FeatureExtractor:
    '''Extract features from raw data'''
    
    def extract_features(self, data: pd.DataFrame):
        '''Extract different types of features'''
        features = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'text': []
        }
        
        for col in data.columns:
            dtype = data[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                features['numerical'].append(col)
            elif pd.api.types.is_datetime64_dtype(dtype):
                features['datetime'].append(col)
            elif pd.api.types.is_string_dtype(dtype) or dtype == 'object':
                # Check if categorical or text
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.5:  # Likely categorical
                    features['categorical'].append(col)
                else:  # Likely text
                    features['text'].append(col)
        
        return features

class FeatureTransformer:
    '''Transform features'''
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
    
    def transform_features(self, data: pd.DataFrame, 
                          numerical_features: List[str],
                          categorical_features: List[str]):
        '''Apply transformations to features'''
        transformed_data = data.copy()
        counts = {
            'scaled_count': 0,
            'encoded_count': 0,
            'binned_count': 0,
            'log_transformed_count': 0
        }
        
        # Scale numerical features
        if numerical_features:
            for col in numerical_features:
                if col in transformed_data.columns:
                    # Standard scaling
                    transformed_data[f'{col}_scaled'] = self.scalers['standard'].fit_transform(
                        transformed_data[[col]]
                    )
                    counts['scaled_count'] += 1
                    
                    # Log transformation for skewed features
                    if abs(transformed_data[col].skew()) > 1:
                        if (transformed_data[col] > 0).all():
                            transformed_data[f'{col}_log'] = np.log1p(transformed_data[col])
                            counts['log_transformed_count'] += 1
        
        # Encode categorical features
        if categorical_features:
            for col in categorical_features:
                if col in transformed_data.columns:
                    # One-hot encoding for low cardinality
                    if transformed_data[col].nunique() <= 10:
                        dummies = pd.get_dummies(transformed_data[col], prefix=col)
                        transformed_data = pd.concat([transformed_data, dummies], axis=1)
                        counts['encoded_count'] += transformed_data[col].nunique()
                    else:
                        # Label encoding for high cardinality
                        transformed_data[f'{col}_encoded'] = pd.factorize(transformed_data[col])[0]
                        counts['encoded_count'] += 1
        
        # Binning for continuous features
        for col in numerical_features[:5]:  # Limit to first 5 features
            if col in transformed_data.columns:
                transformed_data[f'{col}_binned'] = pd.qcut(
                    transformed_data[col], 
                    q=5, 
                    labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                    duplicates='drop'
                )
                counts['binned_count'] += 1
        
        return {
            'data': transformed_data,
            **counts
        }

class FeatureGenerator:
    '''Generate new features'''
    
    def generate_polynomial_features(self, data: pd.DataFrame, degree: int = 2):
        '''Generate polynomial features'''
        poly_features = pd.DataFrame()
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols[:5]:  # Limit to prevent explosion
            for d in range(2, degree + 1):
                poly_features[f'{col}_pow{d}'] = data[col] ** d
        
        return poly_features
    
    def generate_ratio_features(self, data: pd.DataFrame):
        '''Generate ratio features'''
        ratio_features = pd.DataFrame()
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Generate ratios for meaningful pairs
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols[:10]):
                for col2 in numerical_cols[i+1:min(i+3, len(numerical_cols))]:
                    if (data[col2] != 0).all():
                        ratio_features[f'{col1}_div_{col2}'] = data[col1] / data[col2]
        
        return ratio_features

class TechnicalFeatureGenerator:
    '''Generate technical indicators'''
    
    def generate(self, data: pd.DataFrame):
        '''Generate technical features'''
        features = pd.DataFrame(index=data.index)
        
        # Check for price columns
        if 'close' not in data.columns:
            return features
        
        close = data['close']
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = close.rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [12, 26]:
            features[f'ema_{period}'] = close.ewm(span=period).mean()
        
        # RSI
        features['rsi'] = self._calculate_rsi(close, 14)
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        features['bb_upper'] = sma20 + (std20 * 2)
        features['bb_lower'] = sma20 - (std20 * 2)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (close - features['bb_lower']) / features['bb_width']
        
        # Volume indicators if available
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['vwap'] = (close * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Price features
        if 'high' in data.columns and 'low' in data.columns:
            features['high_low_ratio'] = data['high'] / data['low']
            features['daily_range'] = data['high'] - data['low']
            features['daily_range_pct'] = features['daily_range'] / close
        
        # Returns
        features['returns_1d'] = close.pct_change(1)
        features['returns_5d'] = close.pct_change(5)
        features['returns_20d'] = close.pct_change(20)
        
        # Volatility
        features['volatility_20d'] = features['returns_1d'].rolling(window=20).std()
        
        return features
    
    def _calculate_rsi(self, prices, period=14):
        '''Calculate Relative Strength Index'''
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class StatisticalFeatureGenerator:
    '''Generate statistical features'''
    
    def generate(self, data: pd.DataFrame):
        '''Generate statistical features'''
        features = pd.DataFrame(index=data.index)
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols[:20]:  # Limit features
            # Rolling statistics
            for window in [7, 14, 30]:
                features[f'{col}_rolling_mean_{window}'] = data[col].rolling(window).mean()
                features[f'{col}_rolling_std_{window}'] = data[col].rolling(window).std()
                features[f'{col}_rolling_min_{window}'] = data[col].rolling(window).min()
                features[f'{col}_rolling_max_{window}'] = data[col].rolling(window).max()
            
            # Expanding statistics
            features[f'{col}_expanding_mean'] = data[col].expanding().mean()
            features[f'{col}_expanding_std'] = data[col].expanding().std()
            
            # Lag features
            for lag in [1, 3, 7]:
                features[f'{col}_lag_{lag}'] = data[col].shift(lag)
            
            # Difference features
            features[f'{col}_diff_1'] = data[col].diff(1)
            features[f'{col}_diff_7'] = data[col].diff(7)
            
            # Z-score
            features[f'{col}_zscore'] = (data[col] - data[col].mean()) / data[col].std()
            
            # Percentile rank
            features[f'{col}_pct_rank'] = data[col].rank(pct=True)
        
        return features

class TemporalFeatureGenerator:
    '''Generate temporal features'''
    
    def generate(self, data: pd.DataFrame, datetime_columns: List[str]):
        '''Generate temporal features'''
        features = pd.DataFrame(index=data.index)
        
        for col in datetime_columns:
            if col in data.columns:
                dt_series = pd.to_datetime(data[col])
                
                # Basic temporal features
                features[f'{col}_year'] = dt_series.dt.year
                features[f'{col}_month'] = dt_series.dt.month
                features[f'{col}_day'] = dt_series.dt.day
                features[f'{col}_dayofweek'] = dt_series.dt.dayofweek
                features[f'{col}_hour'] = dt_series.dt.hour
                features[f'{col}_minute'] = dt_series.dt.minute
                
                # Cyclical encoding
                features[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                features[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                features[f'{col}_day_sin'] = np.sin(2 * np.pi * dt_series.dt.day / 31)
                features[f'{col}_day_cos'] = np.cos(2 * np.pi * dt_series.dt.day / 31)
                features[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
                features[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
                
                # Special periods
                features[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
                features[f'{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
                features[f'{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
                features[f'{col}_is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
                features[f'{col}_is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
                
                # Time since epoch
                features[f'{col}_timestamp'] = dt_series.astype(np.int64) // 10**9
        
        return features

class TextFeatureGenerator:
    '''Generate text features'''
    
    def generate(self, data: pd.DataFrame, text_columns: List[str]):
        '''Generate text features'''
        features = pd.DataFrame(index=data.index)
        
        for col in text_columns:
            if col in data.columns:
                # Basic text statistics
                features[f'{col}_length'] = data[col].str.len()
                features[f'{col}_word_count'] = data[col].str.split().str.len()
                features[f'{col}_char_count'] = data[col].str.replace(' ', '').str.len()
                
                # Special character counts
                features[f'{col}_digit_count'] = data[col].str.count(r'\d')
                features[f'{col}_upper_count'] = data[col].str.count(r'[A-Z]')
                features[f'{col}_lower_count'] = data[col].str.count(r'[a-z]')
                features[f'{col}_special_count'] = data[col].str.count(r'[^A-Za-z0-9\s]')
                
                # Sentiment indicators (simplified)
                positive_words = ['good', 'great', 'excellent', 'positive', 'up']
                negative_words = ['bad', 'poor', 'negative', 'down', 'loss']
                
                features[f'{col}_positive_words'] = data[col].str.count('|'.join(positive_words))
                features[f'{col}_negative_words'] = data[col].str.count('|'.join(negative_words))
        
        return features

class InteractionFeatureGenerator:
    '''Generate interaction features'''
    
    def generate(self, data: pd.DataFrame, max_interactions: int = 10):
        '''Generate interaction features between columns'''
        features = pd.DataFrame(index=data.index)
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            return features
        
        # Limit columns to prevent explosion
        cols_to_use = numerical_cols[:min(10, len(numerical_cols))]
        
        interaction_count = 0
        for i, col1 in enumerate(cols_to_use):
            for col2 in cols_to_use[i+1:]:
                if interaction_count >= max_interactions:
                    break
                    
                # Multiplication interaction
                features[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                interaction_count += 1
                
                # Division interaction (if no zeros)
                if (data[col2] != 0).all() and interaction_count < max_interactions:
                    features[f'{col1}_div_{col2}'] = data[col1] / data[col2]
                    interaction_count += 1
                
                # Addition interaction
                if interaction_count < max_interactions:
                    features[f'{col1}_plus_{col2}'] = data[col1] + data[col2]
                    interaction_count += 1
            
            if interaction_count >= max_interactions:
                break
        
        return features

class FeatureSelector:
    '''Select best features'''
    
    def __init__(self):
        self.selection_methods = {
            'mutual_info': self._mutual_info_selection,
            'variance': self._variance_selection,
            'correlation': self._correlation_selection,
            'importance': self._importance_selection,
            'pca': self._pca_selection
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50):
        '''Select k best features'''
        
        # Remove any non-numeric columns for selection
        X_numeric = X.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            return {
                'selected_features': X.columns.tolist(),
                'method': 'none',
                'feature_scores': []
            }
        
        # Try multiple selection methods and combine results
        scores = {}
        
        # Mutual information
        mi_scores = self._mutual_info_selection(X_numeric, y, k)
        for feat, score in mi_scores:
            scores[feat] = scores.get(feat, 0) + score
        
        # Variance threshold
        var_scores = self._variance_selection(X_numeric, k)
        for feat, score in var_scores:
            scores[feat] = scores.get(feat, 0) + score
        
        # Correlation with target
        if y is not None:
            corr_scores = self._correlation_selection(X_numeric, y, k)
            for feat, score in corr_scores:
                scores[feat] = scores.get(feat, 0) + score
        
        # Average scores and select top k
        averaged_scores = [(feat, score/3) for feat, score in scores.items()]
        averaged_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = [feat for feat, _ in averaged_scores[:k]]
        
        return {
            'selected_features': selected,
            'method': 'ensemble',
            'feature_scores': averaged_scores
        }
    
    def _mutual_info_selection(self, X, y, k):
        '''Select using mutual information'''
        if y is None:
            return []
        
        # Handle missing values
        X_filled = X.fillna(X.mean())
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_filled, y) if y.dtype == 'object' else mutual_info_classif(X_filled, y)
        
        # Create feature-score pairs
        feature_scores = [(feat, score) for feat, score in zip(X.columns, mi_scores)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:k]
    
    def _variance_selection(self, X, k):
        '''Select features with highest variance'''
        variances = X.var()
        
        feature_scores = [(feat, var) for feat, var in zip(X.columns, variances)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:k]
    
    def _correlation_selection(self, X, y, k):
        '''Select features with highest correlation to target'''
        correlations = X.corrwith(y).abs()
        
        feature_scores = [(feat, corr) for feat, corr in zip(X.columns, correlations) if not pd.isna(corr)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:k]
    
    def _importance_selection(self, X, y, k):
        '''Select using feature importance from tree model'''
        # Simplified - would use actual tree model
        scores = np.random.uniform(0, 1, len(X.columns))
        
        feature_scores = [(feat, score) for feat, score in zip(X.columns, scores)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:k]
    
    def _pca_selection(self, X, k):
        '''Select using PCA components'''
        pca = PCA(n_components=min(k, len(X.columns), len(X)))
        pca.fit(X.fillna(X.mean()))
        
        # Get feature importance from PCA components
        importance = np.abs(pca.components_).mean(axis=0)
        
        feature_scores = [(feat, imp) for feat, imp in zip(X.columns, importance)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        return feature_scores[:k]

class FeatureStore:
    '''Store and retrieve feature sets'''
    
    def __init__(self):
        self.storage = {}
        self.metadata = {}
        self.lineage = defaultdict(list)
    
    def store_features(self, feature_set: FeatureSet):
        '''Store feature set'''
        # Store features
        self.storage[feature_set.feature_set_id] = feature_set
        
        # Store metadata
        self.metadata[feature_set.feature_set_id] = {
            'created_date': feature_set.created_date,
            'version': feature_set.version,
            'feature_count': len(feature_set.features),
            'feature_types': self._count_feature_types(feature_set)
        }
        
        # Track lineage
        self.lineage[feature_set.name].append({
            'id': feature_set.feature_set_id,
            'version': feature_set.version,
            'timestamp': datetime.now()
        })
        
        return {
            'feature_set_id': feature_set.feature_set_id,
            'location': f'feature_store/{feature_set.feature_set_id}',
            'version': feature_set.version,
            'status': 'stored'
        }
    
    def retrieve_features(self, feature_set_id: str):
        '''Retrieve feature set'''
        return self.storage.get(feature_set_id)
    
    def get_latest_version(self, feature_set_name: str):
        '''Get latest version of feature set'''
        if feature_set_name in self.lineage:
            versions = self.lineage[feature_set_name]
            if versions:
                latest = max(versions, key=lambda x: x['timestamp'])
                return self.storage.get(latest['id'])
        return None
    
    def _count_feature_types(self, feature_set):
        '''Count feature types in set'''
        type_counts = defaultdict(int)
        for feature in feature_set.features:
            type_counts[feature.feature_type.value] += 1
        return dict(type_counts)

class FeatureValidator:
    '''Validate feature quality'''
    
    def assess_quality(self, data: pd.DataFrame):
        '''Assess data quality'''
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        
        # Check for duplicates
        duplicate_rows = data.duplicated().sum()
        
        # Check for outliers (simplified)
        outliers = 0
        for col in data.select_dtypes(include=[np.number]).columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers += ((data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)).sum()
        
        # Data types
        data_types = {
            'numerical': len(data.select_dtypes(include=[np.number]).columns),
            'categorical': len(data.select_dtypes(include=['object']).columns),
            'datetime': len(data.select_dtypes(include=['datetime']).columns)
        }
        
        # Overall quality
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        if missing_pct < 5 and duplicate_rows < data.shape[0] * 0.01:
            quality = DataQuality.EXCELLENT
        elif missing_pct < 10 and duplicate_rows < data.shape[0] * 0.05:
            quality = DataQuality.GOOD
        elif missing_pct < 20:
            quality = DataQuality.FAIR
        elif missing_pct < 50:
            quality = DataQuality.POOR
        else:
            quality = DataQuality.UNUSABLE
        
        return {
            'overall_quality': quality,
            'missing_percentage': missing_pct,
            'duplicate_rows': duplicate_rows,
            'outlier_percentage': (outliers / total_cells) * 100 if total_cells > 0 else 0,
            'data_types': data_types,
            'shape': data.shape
        }
    
    def validate_features(self, features: pd.DataFrame):
        '''Validate features'''
        warnings = []
        invalid_features = []
        
        for col in features.columns:
            # Check for constant features
            if features[col].nunique() == 1:
                invalid_features.append(col)
                warnings.append(f'{col} is constant')
            
            # Check for high missing ratio
            missing_ratio = features[col].isnull().sum() / len(features)
            if missing_ratio > 0.5:
                warnings.append(f'{col} has {missing_ratio:.1%} missing values')
            
            # Check for high correlation (simplified)
            if col in features.select_dtypes(include=[np.number]).columns:
                correlations = features.select_dtypes(include=[np.number]).corrwith(features[col]).abs()
                high_corr = correlations[correlations > 0.95]
                if len(high_corr) > 1:  # More than just self-correlation
                    warnings.append(f'{col} highly correlated with {len(high_corr)-1} features')
        
        return {
            'valid_count': len(features.columns) - len(invalid_features),
            'invalid_count': len(invalid_features),
            'invalid_features': invalid_features,
            'warnings': warnings[:10]  # Limit warnings
        }

# Demonstrate the system
if __name__ == '__main__':
    print('🔧 FEATURE ENGINEERING PIPELINE - ULTRAPLATFORM')
    print('='*80)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='D')
    
    sample_data = pd.DataFrame({
        'date': dates,
        'close': 100 + np.cumsum(np.random.randn(1000) * 2),
        'open': 100 + np.cumsum(np.random.randn(1000) * 2),
        'high': 105 + np.cumsum(np.random.randn(1000) * 2),
        'low': 95 + np.cumsum(np.random.randn(1000) * 2),
        'volume': np.random.randint(1000000, 10000000, 1000),
        'volatility': np.abs(np.random.randn(1000) * 0.02),
        'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 1000),
        'news_text': ['Market moves ' + np.random.choice(['up', 'down']) for _ in range(1000)],
        'target': np.random.choice([0, 1], 1000)  # Binary target
    })
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline()
    
    # Process data
    print('\n🎯 PROCESSING DATA THROUGH PIPELINE')
    print('='*80 + '\n')
    
    result = pipeline.process_data(
        data=sample_data,
        target='target'
    )
    
    # Show results
    print('\n' + '='*80)
    print('FEATURE ENGINEERING RESULTS')
    print('='*80)
    print(f'Final Feature Count: {result["data"].shape[1]}')
    print(f'Sample Size: {result["data"].shape[0]}')
    print(f'Feature Set ID: {result["feature_set"].feature_set_id}')
    print(f'Data Quality: {result["quality_report"]["overall_quality"].value}')
    
    # Show top features
    print('\nTop Features Generated:')
    for i, feature in enumerate(result["feature_set"].features[:10], 1):
        print(f'  {i}. {feature.name} ({feature.feature_type.value})')
    
    print('\n✅ Feature Engineering Pipeline Operational!')
