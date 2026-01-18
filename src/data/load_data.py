"""
データ読み込みモジュール
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Tuple, Optional


def load_config(config_path: str = "config.yaml") -> dict:
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_demand_data(
    file_path: str,
    date_col: str = "date",
    demand_col: str = "demand"
) -> pd.DataFrame:
    """
    ガス需要データを読み込む
    
    Parameters
    ----------
    file_path : str
        データファイルのパス
    date_col : str
        日付カラム名
    demand_col : str
        需要量カラム名
    
    Returns
    -------
    pd.DataFrame
        需要データ
    """
    df = pd.read_csv(file_path)
    
    # 日付カラムをdatetime型に変換
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    
    # 需要量カラムの確認
    if demand_col not in df.columns:
        raise ValueError(f"需要量カラム '{demand_col}' が見つかりません")
    
    # 日付でソート
    df = df.sort_index()
    
    return df


def load_weather_data(
    file_path: str,
    date_col: str = "date"
) -> pd.DataFrame:
    """
    気象データを読み込む
    
    Parameters
    ----------
    file_path : str
        データファイルのパス
    date_col : str
        日付カラム名
    
    Returns
    -------
    pd.DataFrame
        気象データ
    """
    df = pd.read_csv(file_path)
    
    # 日付カラムをdatetime型に変換
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    
    # 日付でソート
    df = df.sort_index()
    
    return df


def merge_data(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    demand_col: str = "demand"
) -> pd.DataFrame:
    """
    需要データと気象データをマージ
    
    Parameters
    ----------
    demand_df : pd.DataFrame
        需要データ
    weather_df : pd.DataFrame
        気象データ
    demand_col : str
        需要量カラム名
    
    Returns
    -------
    pd.DataFrame
        マージされたデータ
    """
    # インデックス（日付）でマージ
    merged_df = pd.merge(
        demand_df[[demand_col]],
        weather_df,
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    return merged_df


def generate_sample_data(
    start_date: str = "2020-01-01",
    end_date: str = "2024-03-31",
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    サンプルデータを生成（デモ用）
    
    Parameters
    ----------
    start_date : str
        開始日
    end_date : str
        終了日
    save_path : str, optional
        保存パス
    
    Returns
    -------
    pd.DataFrame
        サンプルデータ
    """
    # 日付範囲を生成
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # サンプル需要データを生成（季節性とランダムノイズを含む）
    np.random.seed(42)
    n_days = len(dates)
    
    # ベースライン需要
    base_demand = 1000
    
    # 季節性（冬に高い、夏に低い）
    seasonal = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 - np.pi/2)
    
    # 週次パターン（週末に低い）
    day_of_week = dates.dayofweek
    weekly = -50 * (day_of_week >= 5).astype(int)  # 土日は-50
    
    # ランダムノイズ
    noise = np.random.normal(0, 50, n_days)
    
    # 需要量を計算
    demand = base_demand + seasonal + weekly + noise
    demand = np.maximum(demand, 0)  # 負の値を0に
    
    # 気象データを生成
    # 気温（季節性を含む）
    temp_base = 15
    temp_seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 - np.pi/2)
    temp_noise = np.random.normal(0, 3, n_days)
    temperature = temp_base + temp_seasonal + temp_noise
    
    # 降水量（ランダム）
    precipitation = np.random.exponential(2, n_days)
    precipitation = np.minimum(precipitation, 50)  # 最大50mm
    
    # データフレームを作成
    df = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'temperature': temperature,
        'precipitation': precipitation
    })
    
    df = df.set_index('date')
    
    # 保存
    if save_path:
        df.to_csv(save_path)
        print(f"サンプルデータを {save_path} に保存しました")
    
    return df


if __name__ == "__main__":
    # サンプルデータの生成
    print("サンプルデータを生成中...")
    sample_data = generate_sample_data(
        start_date="2020-01-01",
        end_date="2024-03-31",
        save_path="data/raw/sample_data.csv"
    )
    print(f"生成されたデータ形状: {sample_data.shape}")
    print(sample_data.head())
