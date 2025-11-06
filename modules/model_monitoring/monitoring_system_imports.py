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
# Note: jensen_shannon removed as it's not needed for basic functionality
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
