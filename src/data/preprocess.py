"""
データ前処理モジュール
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "forward_fill"
) -> pd.DataFrame:
    """
    欠損値を処理
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    method : str
        処理方法 ("forward_fill", "backward_fill", "interpolate", "drop")
    
    Returns
    -------
    pd.DataFrame
        処理後のデータフレーム
    """
    df_processed = df.copy()
    
    if method == "forward_fill":
        df_processed = df_processed.ffill()
    elif method == "backward_fill":
        df_processed = df_processed.bfill()
    elif method == "interpolate":
        df_processed = df_processed.interpolate(method='linear')
    elif method == "drop":
        df_processed = df_processed.dropna()
    else:
        raise ValueError(f"未知の処理方法: {method}")
    
    return df_processed


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 3.0
) -> pd.Series:
    """
    外れ値を検出
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    column : str
        対象カラム
    method : str
        検出方法 ("iqr", "zscore")
    threshold : float
        閾値
    
    Returns
    -------
    pd.Series
        外れ値フラグ（True: 外れ値）
    """
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == "zscore":
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold
    
    else:
        raise ValueError(f"未知の検出方法: {method}")
    
    return outliers


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    外れ値を除去
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    column : str
        対象カラム
    method : str
        検出方法
    threshold : float
        閾値
    
    Returns
    -------
    pd.DataFrame
        処理後のデータフレーム
    """
    outliers = detect_outliers(df, column, method, threshold)
    df_processed = df[~outliers].copy()
    
    return df_processed


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    時間特徴量を作成
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム（日付がインデックス）
    
    Returns
    -------
    pd.DataFrame
        時間特徴量を追加したデータフレーム
    """
    df_processed = df.copy()
    
    # 日付がインデックスでない場合はエラー
    if not isinstance(df_processed.index, pd.DatetimeIndex):
        raise ValueError("インデックスがDatetimeIndexではありません")
    
    # 基本時間特徴量
    df_processed['year'] = df_processed.index.year
    df_processed['month'] = df_processed.index.month
    df_processed['day'] = df_processed.index.day
    df_processed['day_of_week'] = df_processed.index.dayofweek
    df_processed['day_of_year'] = df_processed.index.dayofyear
    df_processed['week'] = df_processed.index.isocalendar().week
    df_processed['quarter'] = df_processed.index.quarter
    
    # 季節性特徴量（三角関数変換）
    df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
    df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
    df_processed['day_of_year_sin'] = np.sin(2 * np.pi * df_processed['day_of_year'] / 365.25)
    df_processed['day_of_year_cos'] = np.cos(2 * np.pi * df_processed['day_of_year'] / 365.25)
    df_processed['day_of_week_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
    df_processed['day_of_week_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
    
    # 祝日フラグ（簡易版、実際はholidaysライブラリを使用）
    # ここでは土日を祝日として扱う
    df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
    
    return df_processed


def split_time_series(
    df: pd.DataFrame,
    train_end: str,
    val_end: Optional[str] = None,
    test_start: Optional[str] = None
) -> tuple:
    """
    時系列データを訓練・検証・テストに分割
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    train_end : str
        訓練期間の終了日
    val_end : str, optional
        検証期間の終了日
    test_start : str, optional
        テスト期間の開始日
    
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    train_df = df[df.index <= train_end].copy()
    
    if val_end is not None:
        val_df = df[(df.index > train_end) & (df.index <= val_end)].copy()
    else:
        val_df = pd.DataFrame()
    
    if test_start is not None:
        test_df = df[df.index > test_start].copy()
    else:
        test_df = pd.DataFrame()
    
    return train_df, val_df, test_df
