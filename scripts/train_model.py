"""
モデル訓練スクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml
from src.data.load_data import load_config, generate_sample_data
from src.data.preprocess import create_time_features, split_time_series, handle_missing_values
from src.data.feature_engineering import create_all_features
from src.models.train import train_xgboost, train_random_forest, save_model
from src.models.predict import predict_xgboost, predict_random_forest
from src.models.evaluate import calculate_metrics, print_metrics
from src.utils.helpers import ensure_dir


def main():
    """メイン処理"""
    print("=" * 60)
    print("ガス需要予測モデル訓練スクリプト")
    print("=" * 60)
    
    # 設定を読み込む
    config = load_config("config.yaml")
    
    # ディレクトリを作成
    ensure_dir("data/raw")
    ensure_dir("data/processed")
    ensure_dir("models")
    ensure_dir("results/figures")
    
    # データの読み込み（サンプルデータを生成）
    print("\n[1/6] データの準備...")
    try:
        # 実際のデータがある場合は読み込む
        df = pd.read_csv("data/raw/gas_demand.csv", index_col=0, parse_dates=True)
        print("実際のデータを読み込みました")
    except:
        # サンプルデータを生成
        print("サンプルデータを生成します...")
        df = generate_sample_data(
            start_date=config['model']['train_start_date'],
            end_date=config['model']['test_end_date']
        )
        df.to_csv("data/raw/sample_data.csv")
        print("サンプルデータを生成しました")
    
    print(f"データ形状: {df.shape}")
    print(f"期間: {df.index.min()} ～ {df.index.max()}")
    
    # 前処理
    print("\n[2/6] データの前処理...")
    df = handle_missing_values(df, method="forward_fill")
    df = create_time_features(df)
    print("前処理が完了しました")
    
    # 特徴量エンジニアリング
    print("\n[3/6] 特徴量エンジニアリング...")
    df = create_all_features(
        df,
        target_col="demand",
        lags=config['features']['lags'],
        ma_windows=config['features']['moving_averages']
    )
    
    # 欠損値を除去（特徴量作成で生じた欠損値）
    df = df.dropna()
    print(f"特徴量作成後のデータ形状: {df.shape}")
    print(f"特徴量数: {len(df.columns) - 1}")  # demandを除く
    
    # 処理済みデータを保存
    ensure_dir("data/processed")
    df.to_csv("data/processed/features.csv")
    print("処理済みデータを data/processed/features.csv に保存しました")
    
    # データ分割
    print("\n[4/6] データ分割...")
    train_df, val_df, test_df = split_time_series(
        df,
        train_end=config['model']['train_end_date'],
        val_end=config['model']['val_end_date'],
        test_start=config['model']['test_start_date']
    )
    
    print(f"訓練データ: {len(train_df)} サンプル")
    print(f"検証データ: {len(val_df)} サンプル")
    print(f"テストデータ: {len(test_df)} サンプル")
    
    # 特徴量とターゲットを分離
    target_col = "demand"
    feature_cols = [col for col in df.columns if col != target_col]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols] if len(val_df) > 0 else None
    y_val = val_df[target_col] if len(val_df) > 0 else None
    X_test = test_df[feature_cols] if len(test_df) > 0 else None
    y_test = test_df[target_col] if len(test_df) > 0 else None
    
    # モデル訓練
    print("\n[5/6] モデル訓練...")
    models = {}
    
    # XGBoost
    print("XGBoostモデルを訓練中...")
    xgb_model = train_xgboost(
        X_train, y_train,
        X_val, y_val,
        params=config['model']['xgboost']
    )
    models['xgboost'] = xgb_model
    
    # 検証データで評価
    if X_val is not None and y_val is not None:
        y_pred_val = predict_xgboost(xgb_model, X_val)
        metrics_val = calculate_metrics(y_val.values, y_pred_val)
        print("\n検証データでの評価:")
        print_metrics(metrics_val)
    
    # Random Forest
    print("\nRandom Forestモデルを訓練中...")
    rf_model = train_random_forest(
        X_train, y_train,
        params=config['model']['random_forest']
    )
    models['random_forest'] = rf_model
    
    # モデル保存
    print("\n[6/6] モデルの保存...")
    for model_name, model in models.items():
        save_model(model, f"models/{model_name}_model.pkl")
    
    # テストデータで評価
    if X_test is not None and y_test is not None:
        print("\nテストデータでの評価:")
        for model_name, model in models.items():
            print(f"\n{model_name.upper()}モデル:")
            if model_name == 'xgboost':
                y_pred_test = predict_xgboost(model, X_test)
            elif model_name == 'random_forest':
                y_pred_test = predict_random_forest(model, X_test)
            
            metrics_test = calculate_metrics(y_test.values, y_pred_test)
            print_metrics(metrics_test)
    
    print("\n" + "=" * 60)
    print("モデル訓練が完了しました！")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. SHAP分析を実行: python scripts/interpretability_analysis.py")
    print("2. ダッシュボード起動: streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
