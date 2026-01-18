"""
解釈可能性分析スクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml
from src.data.load_data import load_config
from src.models.train import load_model
from src.models.predict import predict_xgboost, predict_random_forest
from src.interpretability.shap_analysis import (
    calculate_shap_values,
    plot_summary,
    plot_dependence,
    get_feature_importance
)
from src.interpretability.visualization import (
    plot_feature_importance,
    plot_time_series
)
from src.utils.helpers import ensure_dir


def main():
    """メイン処理"""
    print("=" * 60)
    print("解釈可能性分析スクリプト")
    print("=" * 60)
    
    # 設定を読み込む
    config = load_config("config.yaml")
    
    # ディレクトリを作成
    ensure_dir("results/figures")
    
    # データの読み込み
    print("\n[1/4] データの読み込み...")
    try:
        df = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
    except:
        print("エラー: 処理済みデータが見つかりません。先にtrain_model.pyを実行してください。")
        return
    
    # テストデータを準備
    target_col = "demand"
    feature_cols = [col for col in df.columns if col != target_col]
    
    # テスト期間のデータを取得
    test_start = config['model']['test_start_date']
    test_df = df[df.index >= test_start].copy()
    
    if len(test_df) == 0:
        print("エラー: テストデータが見つかりません。")
        return
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"テストデータ: {len(X_test)} サンプル")
    
    # モデルの読み込み
    print("\n[2/4] モデルの読み込み...")
    model_name = "xgboost"  # または "random_forest"
    model = load_model(f"models/{model_name}_model.pkl")
    print(f"{model_name.upper()}モデルを読み込みました")
    
    # 予測
    print("\n[3/4] 予測の実行...")
    if model_name == 'xgboost':
        y_pred = predict_xgboost(model, X_test)
    elif model_name == 'random_forest':
        y_pred = predict_random_forest(model, X_test)
    
    # SHAP分析
    print("\n[4/4] SHAP分析の実行...")
    sample_size = config['interpretability']['shap']['sample_size']
    
    explainer, shap_values, X_sample = calculate_shap_values(
        model, X_test, sample_size=sample_size
    )
    
    print(f"SHAP値を計算しました（サンプル数: {len(X_sample)}）")
    
    # 可視化
    print("\n可視化を生成中...")
    
    # Summary Plot
    plot_summary(
        shap_values,
        X_sample,
        save_path="results/figures/shap_summary.png"
    )
    
    # Dependence Plot（気温）
    if 'temperature' in X_sample.columns:
        plot_dependence(
            shap_values,
            X_sample,
            feature='temperature',
            save_path="results/figures/shap_dependence_temperature.png"
        )
    
    # 特徴量重要度
    importance_df = get_feature_importance(shap_values, X_sample.columns.tolist())
    importance_df.to_csv("results/feature_importance.csv", index=False)
    
    plot_feature_importance(
        importance_df,
        top_n=20,
        save_path="results/figures/feature_importance.png"
    )
    
    # 時系列プロット（予測値と実測値）
    plot_df = test_df.copy()
    plot_df['prediction'] = y_pred
    plot_time_series(
        plot_df,
        target_col="demand",
        pred_col="prediction",
        save_path="results/figures/time_series_prediction.png"
    )
    
    print("\n" + "=" * 60)
    print("解釈可能性分析が完了しました！")
    print("結果は results/figures/ に保存されています。")
    print("=" * 60)
    
    # 特徴量重要度トップ10を表示
    print("\n特徴量重要度トップ10:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
