"""
SHAP分析モジュール
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Optional


def _predict_wrapper(model, X):
    """XGBoostモデルの予測ラッパー（SHAP互換性のため）"""
    if hasattr(model, 'predict'):
        return model.predict(X)
    else:
        raise ValueError("Model does not have predict method")


def calculate_shap_values(
    model: Any,
    X: pd.DataFrame,
    sample_size: Optional[int] = None
) -> tuple:
    """
    SHAP値を計算
    
    Parameters
    ----------
    model : Any
        訓練済みモデル（XGBoost, Random Forest等）
    X : pd.DataFrame
        特徴量データ
    sample_size : int, optional
        サンプルサイズ（計算を高速化するため）
    
    Returns
    -------
    tuple
        (explainer, shap_values, X_sample)
    """
    # サンプリング
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    # TreeSHAP（XGBoost, Random Forest用）
    try:
        # XGBoostの新しいバージョンに対応
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        # TreeSHAPが使えない場合はKernelSHAP（ラッパー関数を使用）
        print(f"TreeSHAPが使用できません。KernelSHAPを使用します: {str(e)[:100]}")
        background_size = min(50, len(X_sample))
        background = X_sample.sample(n=background_size, random_state=42)
        
        # ラッパー関数を作成
        def predict_func(X_input):
            if isinstance(X_input, pd.DataFrame):
                return model.predict(X_input)
            else:
                # numpy配列の場合
                X_df = pd.DataFrame(X_input, columns=X_sample.columns)
                return model.predict(X_df)
        
        explainer = shap.KernelExplainer(predict_func, background.values)
        shap_values = explainer.shap_values(X_sample.values)
    
    return explainer, shap_values, X_sample


def plot_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: Optional[str] = None,
    max_display: int = 20
):
    """
    SHAP Summary Plotを描画
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP値
    X : pd.DataFrame
        特徴量データ
    save_path : str, optional
        保存パス
    max_display : int
        表示する特徴量の最大数
    """
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary Plotを {save_path} に保存しました")
    
    plt.close()


def plot_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    SHAP Dependence Plotを描画
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP値
    X : pd.DataFrame
        特徴量データ
    feature : str
        対象特徴量
    interaction_feature : str, optional
        相互作用特徴量
    save_path : str, optional
        保存パス
    """
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_values,
        X,
        interaction_index=interaction_feature,
        show=False
    )
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dependence Plotを {save_path} に保存しました")
    
    plt.close()


def plot_waterfall(
    explainer: Any,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    instance_idx: int,
    save_path: Optional[str] = None
):
    """
    SHAP Waterfall Plotを描画
    
    Parameters
    ----------
    explainer : Any
        SHAP explainer
    shap_values : np.ndarray
        SHAP値
    X : pd.DataFrame
        特徴量データ
    instance_idx : int
        インスタンスのインデックス
    save_path : str, optional
        保存パス
    """
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        explainer(X.iloc[[instance_idx]]),
        show=False
    )
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Waterfall Plotを {save_path} に保存しました")
    
    plt.close()


def get_feature_importance(
    shap_values: np.ndarray,
    feature_names: list
) -> pd.DataFrame:
    """
    特徴量重要度を計算（SHAP値の絶対値の平均）
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP値
    feature_names : list
        特徴量名のリスト
    
    Returns
    -------
    pd.DataFrame
        特徴量重要度
    """
    # SHAP値の絶対値の平均
    importance = np.abs(shap_values).mean(axis=0)
    
    # データフレームに変換
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df
