"""
可視化モジュール
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


def plot_time_series(
    df: pd.DataFrame,
    target_col: str = "demand",
    pred_col: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    時系列プロット
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム（日付がインデックス）
    target_col : str
        実測値カラム名
    pred_col : str, optional
        予測値カラム名
    save_path : str, optional
        保存パス
    """
    plt.figure(figsize=(14, 6))
    
    plt.plot(df.index, df[target_col], label='実測値', alpha=0.7)
    
    if pred_col and pred_col in df.columns:
        plt.plot(df.index, df[pred_col], label='予測値', alpha=0.7)
    
    plt.xlabel('日付')
    plt.ylabel('需要量')
    plt.title('ガス需要の時系列')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"時系列プロットを {save_path} に保存しました")
    
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    残差プロット
    
    Parameters
    ----------
    y_true : np.ndarray
        実測値
    y_pred : np.ndarray
        予測値
    save_path : str, optional
        保存パス
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 残差の時系列プロット
    axes[0].plot(residuals)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('サンプル')
    axes[0].set_ylabel('残差')
    axes[0].set_title('残差の時系列')
    axes[0].grid(True, alpha=0.3)
    
    # 残差のヒストグラム
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('残差')
    axes[1].set_ylabel('頻度')
    axes[1].set_title('残差の分布')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"残差プロットを {save_path} に保存しました")
    
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None
):
    """
    特徴量重要度プロット
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        特徴量重要度のデータフレーム
    top_n : int
        表示する上位N個
    save_path : str, optional
        保存パス
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('重要度')
    plt.title(f'特徴量重要度（上位{top_n}個）')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特徴量重要度プロットを {save_path} に保存しました")
    
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    相関ヒートマップ
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    save_path : str, optional
        保存パス
    """
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(
        correlation,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title('特徴量間の相関')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相関ヒートマップを {save_path} に保存しました")
    
    plt.close()
