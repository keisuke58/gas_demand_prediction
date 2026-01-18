"""
モデル訓練モジュール
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
import yaml


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict[str, Any]] = None
) -> xgb.XGBRegressor:
    """
    XGBoostモデルを訓練
    
    Parameters
    ----------
    X_train : pd.DataFrame
        訓練データの特徴量
    y_train : pd.Series
        訓練データのターゲット
    X_val : pd.DataFrame, optional
        検証データの特徴量
    y_val : pd.Series, optional
        検証データのターゲット
    params : dict, optional
        ハイパーパラメータ
    
    Returns
    -------
    xgb.XGBRegressor
        訓練済みモデル
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    
    # 検証データがある場合はearly stoppingを有効化
    if X_val is not None and y_val is not None:
        # 新しいXGBoostではearly_stopping_roundsをparamsに含める
        if 'early_stopping_rounds' not in params:
            params['early_stopping_rounds'] = 10
    
    model = xgb.XGBRegressor(**params)
    
    # 検証データがある場合はeval_setを指定
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> RandomForestRegressor:
    """
    Random Forestモデルを訓練
    
    Parameters
    ----------
    X_train : pd.DataFrame
        訓練データの特徴量
    y_train : pd.Series
        訓練データのターゲット
    params : dict, optional
        ハイパーパラメータ
    
    Returns
    -------
    RandomForestRegressor
        訓練済みモデル
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def train_prophet(
    df: pd.DataFrame,
    date_col: str = "ds",
    target_col: str = "y",
    params: Optional[Dict[str, Any]] = None
):
    """
    Prophetモデルを訓練
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム（日付とターゲットを含む）
    date_col : str
        日付カラム名
    target_col : str
        ターゲットカラム名
    params : dict, optional
        ハイパーパラメータ
    
    Returns
    -------
    Any
        訓練済みモデル（Prophet）
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed. Install it with: pip install prophet")
    
    # Prophet用のデータフレームを作成
    prophet_df = df[[date_col, target_col]].copy()
    prophet_df.columns = ['ds', 'y']
    
    if params is None:
        params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
    
    model = Prophet(**params)
    model.fit(prophet_df)
    
    return model


def save_model(model: Any, file_path: str):
    """
    モデルを保存
    
    Parameters
    ----------
    model : Any
        保存するモデル
    file_path : str
        保存パス
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"モデルを {file_path} に保存しました")


def load_model(file_path: str) -> Any:
    """
    モデルを読み込む
    
    Parameters
    ----------
    file_path : str
        モデルファイルのパス
    
    Returns
    -------
    Any
        読み込んだモデル
    """
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    return model
