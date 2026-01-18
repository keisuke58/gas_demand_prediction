"""
特徴量エンジニアリングモジュール
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int] = [1, 7, 30]
) -> pd.DataFrame:
    """
    ラグ特徴量を作成
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    target_col : str
        ターゲットカラム名
    lags : List[int]
        ラグのリスト
    
    Returns
    -------
    pd.DataFrame
        ラグ特徴量を追加したデータフレーム
    """
    df_processed = df.copy()
    
    for lag in lags:
        df_processed[f'{target_col}_lag_{lag}'] = df_processed[target_col].shift(lag)
    
    return df_processed


def create_moving_average_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int] = [7, 30, 90]
) -> pd.DataFrame:
    """
    移動平均特徴量を作成
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    target_col : str
        ターゲットカラム名
    windows : List[int]
        ウィンドウサイズのリスト
    
    Returns
    -------
    pd.DataFrame
        移動平均特徴量を追加したデータフレーム
    """
    df_processed = df.copy()
    
    for window in windows:
        df_processed[f'{target_col}_ma_{window}'] = (
            df_processed[target_col].rolling(window=window).mean()
        )
        df_processed[f'{target_col}_std_{window}'] = (
            df_processed[target_col].rolling(window=window).std()
        )
    
    return df_processed


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int] = [7, 30]
) -> pd.DataFrame:
    """
    ローリング統計特徴量を作成
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    target_col : str
        ターゲットカラム名
    windows : List[int]
        ウィンドウサイズのリスト
    
    Returns
    -------
    pd.DataFrame
        ローリング特徴量を追加したデータフレーム
    """
    df_processed = df.copy()
    
    for window in windows:
        # 最小値、最大値
        df_processed[f'{target_col}_min_{window}'] = (
            df_processed[target_col].rolling(window=window).min()
        )
        df_processed[f'{target_col}_max_{window}'] = (
            df_processed[target_col].rolling(window=window).max()
        )
    
    return df_processed


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    気象特徴量を作成
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム（気温、降水量を含む）
    
    Returns
    -------
    pd.DataFrame
        気象特徴量を追加したデータフレーム
    """
    df_processed = df.copy()
    
    # 気温関連特徴量
    if 'temperature' in df_processed.columns:
        # 累積度日（Cooling Degree Days, Heating Degree Days）
        base_temp = 18.0  # 基準温度
        df_processed['cdd'] = np.maximum(df_processed['temperature'] - base_temp, 0)
        df_processed['hdd'] = np.maximum(base_temp - df_processed['temperature'], 0)
        
        # 気温の差分
        df_processed['temp_diff'] = df_processed['temperature'].diff()
        df_processed['temp_diff_7'] = df_processed['temperature'].diff(7)
        
        # 気温の移動平均
        df_processed['temp_ma_7'] = df_processed['temperature'].rolling(7).mean()
        df_processed['temp_ma_30'] = df_processed['temperature'].rolling(30).mean()
    
    # 降水量関連特徴量
    if 'precipitation' in df_processed.columns:
        # 降水量の移動平均
        df_processed['precip_ma_7'] = df_processed['precipitation'].rolling(7).mean()
        df_processed['precip_ma_30'] = df_processed['precipitation'].rolling(30).mean()
        
        # 雨が降った日かどうか
        df_processed['is_rainy'] = (df_processed['precipitation'] > 0).astype(int)
    
    return df_processed


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    交互作用特徴量を作成
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    
    Returns
    -------
    pd.DataFrame
        交互作用特徴量を追加したデータフレーム
    """
    df_processed = df.copy()
    
    # 気温 × 曜日
    if 'temperature' in df_processed.columns and 'day_of_week' in df_processed.columns:
        df_processed['temp_x_weekday'] = (
            df_processed['temperature'] * df_processed['day_of_week']
        )
    
    # 気温 × 月
    if 'temperature' in df_processed.columns and 'month' in df_processed.columns:
        df_processed['temp_x_month'] = (
            df_processed['temperature'] * df_processed['month']
        )
    
    # 気温 × 週末
    if 'temperature' in df_processed.columns and 'is_weekend' in df_processed.columns:
        df_processed['temp_x_weekend'] = (
            df_processed['temperature'] * df_processed['is_weekend']
        )
    
    return df_processed


def create_all_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    lags: List[int] = [1, 7, 30],
    ma_windows: List[int] = [7, 30, 90]
) -> pd.DataFrame:
    """
    すべての特徴量を作成
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    target_col : str
        ターゲットカラム名
    lags : List[int]
        ラグのリスト
    ma_windows : List[int]
        移動平均のウィンドウサイズのリスト
    
    Returns
    -------
    pd.DataFrame
        すべての特徴量を追加したデータフレーム
    """
    df_processed = df.copy()
    
    # 時間特徴量（preprocess.pyで作成済みと仮定）
    # ラグ特徴量
    df_processed = create_lag_features(df_processed, target_col, lags)
    
    # 移動平均特徴量
    df_processed = create_moving_average_features(df_processed, target_col, ma_windows)
    
    # ローリング特徴量
    df_processed = create_rolling_features(df_processed, target_col, [7, 30])
    
    # 気象特徴量
    df_processed = create_weather_features(df_processed)
    
    # 交互作用特徴量
    df_processed = create_interaction_features(df_processed)
    
    return df_processed
