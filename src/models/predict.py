"""
予測モジュール
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def predict_xgboost(
    model: Any,
    X: pd.DataFrame
) -> np.ndarray:
    """
    XGBoostモデルで予測
    
    Parameters
    ----------
    model : Any
        訓練済みモデル
    X : pd.DataFrame
        特徴量
    
    Returns
    -------
    np.ndarray
        予測値
    """
    predictions = model.predict(X)
    return predictions


def predict_random_forest(
    model: Any,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Random Forestモデルで予測
    
    Parameters
    ----------
    model : Any
        訓練済みモデル
    X : pd.DataFrame
        特徴量
    
    Returns
    -------
    np.ndarray
        予測値
    """
    predictions = model.predict(X)
    return predictions


def predict_prophet(
    model: Any,
    periods: int,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Prophetモデルで予測
    
    Parameters
    ----------
    model : Any
        訓練済みモデル（Prophet）
    periods : int
        予測期間（日数）
    freq : str
        頻度（'D': 日次）
    
    Returns
    -------
    pd.DataFrame
        予測結果（ds, yhat, yhat_lower, yhat_upper）
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed. Install it with: pip install prophet")
    
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast
