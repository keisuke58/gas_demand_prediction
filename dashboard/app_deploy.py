"""
Streamlitダッシュボード（デプロイ用）
モデルとデータが存在しない場合は自動生成
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.models.train import load_model, train_xgboost, train_random_forest, save_model
from src.models.predict import predict_xgboost, predict_random_forest
from src.interpretability.shap_analysis import calculate_shap_values, get_feature_importance
from src.data.load_data import generate_sample_data, load_config
from src.data.preprocess import create_time_features, handle_missing_values
from src.data.feature_engineering import create_all_features
from src.models.evaluate import calculate_metrics
import yaml
import os


# ページ設定
st.set_page_config(
    page_title="ガス需要予測ダッシュボード",
    page_icon="📊",
    layout="wide"
)

# タイトル
st.title("📊 ガス需要予測ダッシュボード")
st.markdown("---")

# サイドバー
st.sidebar.header("設定")

# モデル選択
model_name = st.sidebar.selectbox(
    "モデルを選択",
    ["xgboost", "random_forest"]
)

# データとモデルの準備
@st.cache_data
def prepare_data_and_model():
    """データとモデルを準備（存在しない場合は生成）"""
    config = load_config("config.yaml")
    
    # データの確認と生成
    data_path = "data/processed/features.csv"
    if not os.path.exists(data_path):
        st.info("データを生成中...")
        # サンプルデータを生成
        df = generate_sample_data(
            start_date=config['model']['train_start_date'],
            end_date=config['model']['test_end_date']
        )
        df = handle_missing_values(df, method="forward_fill")
        df = create_time_features(df)
        df = create_all_features(
            df,
            target_col="demand",
            lags=config['features']['lags'],
            ma_windows=config['features']['moving_averages']
        )
        df = df.dropna()
        
        # ディレクトリを作成
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(data_path)
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # モデルの確認と訓練
    model_path = f"models/{model_name}_model.pkl"
    if not os.path.exists(model_path):
        st.info("モデルを訓練中...")
        # データ分割
        from src.data.preprocess import split_time_series
        train_df, val_df, test_df = split_time_series(
            df,
            train_end=config['model']['train_end_date'],
            val_end=config['model']['val_end_date'],
            test_start=config['model']['test_start_date']
        )
        
        target_col = "demand"
        feature_cols = [col for col in df.columns if col != target_col]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols] if len(val_df) > 0 else None
        y_val = val_df[target_col] if len(val_df) > 0 else None
        
        # モデル訓練
        os.makedirs("models", exist_ok=True)
        if model_name == 'xgboost':
            model = train_xgboost(
                X_train, y_train,
                X_val, y_val,
                params=config['model']['xgboost']
            )
        elif model_name == 'random_forest':
            from src.models.train import train_random_forest
            model = train_random_forest(
                X_train, y_train,
                params=config['model']['random_forest']
            )
        
        save_model(model, model_path)
    else:
        model = load_model(model_path)
    
    return df, model

# データとモデルの読み込み
try:
    df, model = prepare_data_and_model()
    
    # メインコンテンツ
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 データ概要")
        st.write(f"**データ期間**: {df.index.min().date()} ～ {df.index.max().date()}")
        st.write(f"**サンプル数**: {len(df):,}")
        st.write(f"**特徴量数**: {len(df.columns) - 1}")
    
    with col2:
        st.subheader("📊 基本統計")
        if "demand" in df.columns:
            st.write(f"**平均需要量**: {df['demand'].mean():.2f}")
            st.write(f"**最大需要量**: {df['demand'].max():.2f}")
            st.write(f"**最小需要量**: {df['demand'].min():.2f}")
    
    st.markdown("---")
    
    # 時系列プロット
    st.subheader("📉 需要の時系列")
    
    # 日付範囲選択
    date_range = st.date_input(
        "表示期間を選択",
        value=(df.index.min().date(), df.index.max().date()),
        min_value=df.index.min().date(),
        max_value=df.index.max().date()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        plot_df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df['demand'],
            mode='lines',
            name='実測値',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="ガス需要の時系列",
            xaxis_title="日付",
            yaxis_title="需要量",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 予測と評価
    st.subheader("🔮 需要予測")
    
    # テスト期間のデータを取得
    try:
        config = load_config("config.yaml")
        test_start = config['model']['test_start_date']
        test_df = df[df.index >= test_start].copy()
        
        if len(test_df) > 0:
            target_col = "demand"
            feature_cols = [col for col in df.columns if col != target_col]
            
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            
            # 予測
            if model_name == 'xgboost':
                y_pred = predict_xgboost(model, X_test)
            elif model_name == 'random_forest':
                y_pred = predict_random_forest(model, X_test)
            
            # 評価指標
            metrics = calculate_metrics(y_test.values, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{metrics['MAE']:.2f}")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
            with col3:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            with col4:
                st.metric("R²", f"{metrics['R2']:.3f}")
            
            # 予測結果のプロット
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=test_df.index,
                y=y_test.values,
                mode='lines',
                name='実測値',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=test_df.index,
                y=y_pred,
                mode='lines',
                name='予測値',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="予測結果（テスト期間）",
                xaxis_title="日付",
                yaxis_title="需要量",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"予測の実行中にエラーが発生しました: {e}")
    
    st.markdown("---")
    
    # 特徴量重要度
    st.subheader("🎯 特徴量重要度（SHAP）")
    
    if st.button("SHAP分析を実行"):
        with st.spinner("SHAP値を計算中..."):
            try:
                # サンプルサイズを制限（計算時間を短縮）
                sample_size = min(50, len(X_test))
                X_sample = X_test.sample(n=sample_size, random_state=42)
                
                explainer, shap_values, X_sample = calculate_shap_values(
                    model, X_sample, sample_size=None
                )
                
                # 特徴量重要度を計算
                importance_df = get_feature_importance(
                    shap_values, X_sample.columns.tolist()
                )
                
                # 可視化
                fig = px.bar(
                    importance_df.head(20),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='特徴量重要度トップ20',
                    labels={'importance': '重要度', 'feature': '特徴量'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # テーブル表示
                st.dataframe(importance_df.head(20), use_container_width=True)
                
            except Exception as e:
                st.error(f"SHAP分析中にエラーが発生しました: {e}")
    
    st.markdown("---")
    
    # データテーブル
    st.subheader("📋 データテーブル")
    
    if st.checkbox("データを表示"):
        st.dataframe(df.tail(100), use_container_width=True)

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
    st.info("データとモデルを準備中です。しばらくお待ちください...")
