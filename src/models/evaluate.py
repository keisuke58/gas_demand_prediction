"""
モデル評価モジュール
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    評価指標を計算
    
    Parameters
    ----------
    y_true : np.ndarray
        実測値
    y_pred : np.ndarray
        予測値
    
    Returns
    -------
    dict
        評価指標の辞書
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    評価指標を表示
    
    Parameters
    ----------
    metrics : dict
        評価指標の辞書
    """
    print("=" * 50)
    print("評価指標")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("=" * 50)
